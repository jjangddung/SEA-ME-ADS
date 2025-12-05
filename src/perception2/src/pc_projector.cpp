#include <memory>
#include <vector>
#include <cmath>
#include <array>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "vision_msgs/msg/detection2_d.hpp"

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

class LanePointProjectorCpp : public rclcpp::Node
{
public:
  LanePointProjectorCpp()
  : Node("pc_projector"),
    fx_(0.0), fy_(0.0), cx_(0.0), cy_(0.0),
    img_w_(0), img_h_(0),
    cam_tz_(0.3), cam_pitch_(0.436332),
    lane_width_(0.35),
    map_received_(false)
  {
    lane_width_ = declare_parameter<double>("lane_width", lane_width_);

    // QoS
    rclcpp::QoS qos_sensor(10);
    qos_sensor.best_effort();

    // 구독
    mask_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "/lane/white_mask", qos_sensor,
      std::bind(&LanePointProjectorCpp::maskCallback, this, std::placeholders::_1));

    caminfo_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "/camera/color/camera_info", qos_sensor,
      std::bind(&LanePointProjectorCpp::camInfoCallback, this, std::placeholders::_1));

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/camera/depth/color/points", qos_sensor,
      std::bind(&LanePointProjectorCpp::cloudCallback, this, std::placeholders::_1));

    yolo_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
      "/yolo/detections_2d", 10,
      std::bind(&LanePointProjectorCpp::yoloCallback, this, std::placeholders::_1));

    pf_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/lane_pf_pose", 10,
      std::bind(&LanePointProjectorCpp::pfCallback, this, std::placeholders::_1));

    // 퍼블리셔
    lane_base_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/camera/depth/lane_points_base_sdf", 10);
    lane_map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/camera/depth/lane_points_map", 10);

    obj_base_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/obstacle/points_base", 10);
    obj_map_pub_  = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/obstacle/points_map", 10);

    RCLCPP_INFO(get_logger(), "pc_projector Started (lane + YOLO unified projector)");
  }

private:
  struct YOLOBox {
    float x1, y1, x2, y2;
    int cls;
    double score;
  };

  /////////////////////////////////////////////////
  // CameraInfo
  /////////////////////////////////////////////////
  void camInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    fx_ = msg->k[0];
    cx_ = msg->k[2];
    fy_ = msg->k[4];
    cy_ = msg->k[5];
  }

  /////////////////////////////////////////////////
  // lane mask (mono8)
  /////////////////////////////////////////////////
  void maskCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try {
      auto cv_ptr = cv_bridge::toCvCopy(msg, "mono8");
      std::lock_guard<std::mutex> lock(mask_mutex_);
      mask_ = cv_ptr->image.clone();
      img_h_ = mask_.rows;
      img_w_ = mask_.cols;
    } catch (const std::exception &e) {
      RCLCPP_ERROR(get_logger(), "maskCallback cv_bridge error: %s", e.what());
      return;
    }
  }

  /////////////////////////////////////////////////
  // YOLO bbox callback
  /////////////////////////////////////////////////
  void yoloCallback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(yolo_mutex_);
    yolo_boxes_.clear();

    for (auto & det : msg->detections)
    {
      int cls = -1;
      double score = 0.0;

      if (!det.results.empty()) {
        const auto & r = det.results[0];

        // Humble: ObjectHypothesisWithPose -> hypothesis.class_id, hypothesis.score
        score = r.hypothesis.score;

        try {
          cls = std::stoi(r.hypothesis.class_id);
        } catch (...) {
          cls = 0;
        }
      }

      const auto & box = det.bbox;
      double cx = box.center.position.x;
      double cy = box.center.position.y;
      double w  = box.size_x;
      double h  = box.size_y;

      double x1 = cx - w * 0.5;
      double y1 = cy - h * 0.5;
      double x2 = cx + w * 0.5;
      double y2 = cy + h * 0.5;

      yolo_boxes_.push_back({(float)x1, (float)y1, (float)x2, (float)y2, cls, score});
    }
  }

  /////////////////////////////////////////////////
  // PF pose (map→base_link)
  /////////////////////////////////////////////////
  void pfCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    double px = msg->pose.pose.position.x;
    double py = msg->pose.pose.position.y;
    double pz = msg->pose.pose.position.z;
    double qx = msg->pose.pose.orientation.x;
    double qy = msg->pose.pose.orientation.y;
    double qz = msg->pose.pose.orientation.z;
    double qw = msg->pose.pose.orientation.w;

    map_R_ = quatToRot(qx,qy,qz,qw);
    map_t_ = {px,py,pz};
    map_received_ = true;
  }

  /////////////////////////////////////////////////
  // Quaternion → Rotation Matrix
  /////////////////////////////////////////////////
  std::array<std::array<double,3>,3>
  quatToRot(double x,double y,double z,double w)
  {
    std::array<std::array<double,3>,3> R;
    double xx=x*x, yy=y*y, zz=z*z, xy=x*y, xz=x*z, yz=y*z, wx=w*x, wy=w*y, wz=w*z;

    R[0][0]=1-2*(yy+zz); R[0][1]=2*(xy-wz);   R[0][2]=2*(xz+wy);
    R[1][0]=2*(xy+wz);   R[1][1]=1-2*(xx+zz); R[1][2]=2*(yz-wx);
    R[2][0]=2*(xz-wy);   R[2][1]=2*(yz+wx);   R[2][2]=1-2*(xx+yy);
    return R;
  }

  /////////////////////////////////////////////////
  // depth PointCloud → lane + object 3D
  /////////////////////////////////////////////////
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (fx_ == 0.0) {
      RCLCPP_DEBUG(get_logger(), "Waiting for camera_info...");
      return;
    }

    cv::Mat mask_local;
    {
      std::lock_guard<std::mutex> lock(mask_mutex_);
      if (mask_.empty()) {
        RCLCPP_DEBUG(get_logger(), "Waiting for lane mask...");
        return;
      }
      mask_local = mask_;  // shallow copy
    }

    int H = img_h_;
    int W = img_w_;

    // YOLO bbox 로컬 복사
    std::vector<YOLOBox> yolo_local;
    {
      std::lock_guard<std::mutex> lock(yolo_mutex_);
      yolo_local = yolo_boxes_;
    }

    // 결과 벡터
    std::vector<std::array<float,3>> lane_base;
    std::vector<std::array<float,3>> lane_map;
    std::vector<std::array<float,3>> obj_base;
    std::vector<std::array<float,3>> obj_map;

    float half_lane = lane_width_ * 0.5f;
    float margin_y  = 0.10f;

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg,"x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg,"y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg,"z");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
    {
      float Xc = *iter_x;
      float Yc = *iter_y;
      float Zc = *iter_z;

      if (Zc <= 0.05f) continue;

      float u = fx_ * (Xc / Zc) + cx_;
      float v = fy_ * (Yc / Zc) + cy_;

      int u_i = static_cast<int>(std::round(u));
      int v_i = static_cast<int>(std::round(v));
      if (u_i < 0 || u_i >= W || v_i < 0 || v_i >= H) continue;

      bool is_lane = (mask_local.at<uint8_t>(v_i,u_i) != 0);

      bool is_object = false;
      for (auto & b : yolo_local) {
        if (u_i >= b.x1 && u_i <= b.x2 && v_i >= b.y1 && v_i <= b.y2) {
          is_object = true;
          break;
        }
      }

      if (!is_lane && !is_object) continue;

      // optical → camera_link
      float Xl = Zc;
      float Yl = -Xc;
      float Zl = -Yc;

      // camera_link → base_link (pitch)
      float Xb = std::cos(cam_pitch_)*Xl + std::sin(cam_pitch_)*Zl;
      float Yb = Yl;
      float Zb = -std::sin(cam_pitch_)*Xl + std::cos(cam_pitch_)*Zl + cam_tz_;

      // lane 범위 필터
      if (is_lane)
      {
        if (Xb < 0.15f || Xb > 1.0f) continue;
        if (std::fabs(Yb) > (half_lane + margin_y)) continue;
      }

      if (is_object)
      {
        if ( Xb > 1.0f) continue;
      }

      if (is_lane) lane_base.push_back({Xb,Yb,Zb});
      if (is_object) obj_base.push_back({Xb,Yb,Zb});

      if (map_received_) {
        double Xm = map_R_[0][0]*Xb + map_R_[0][1]*Yb + map_R_[0][2]*Zb + map_t_[0];
        double Ym = map_R_[1][0]*Xb + map_R_[1][1]*Yb + map_R_[1][2]*Zb + map_t_[1];
        double Zm = map_R_[2][0]*Xb + map_R_[2][1]*Yb + map_R_[2][2]*Zb + map_t_[2];

        if (is_lane) lane_map.push_back({(float)Xm,(float)Ym,(float)Zm});
        if (is_object) obj_map.push_back({(float)Xm,(float)Ym,(float)Zm});
      }
    }

    // lane publish
    {
      sensor_msgs::msg::PointCloud2 cloud;
      cloud.header.stamp = msg->header.stamp;
      cloud.header.frame_id = "base_link";
      createCloud(lane_base, cloud);
      lane_base_pub_->publish(cloud);
    }
    if (map_received_) {
      sensor_msgs::msg::PointCloud2 cloud;
      cloud.header.stamp = msg->header.stamp;
      cloud.header.frame_id = "map";
      createCloud(lane_map, cloud);
      lane_map_pub_->publish(cloud);
    }

    // obstacle publish
    {
      sensor_msgs::msg::PointCloud2 cloud;
      cloud.header.stamp = msg->header.stamp;
      cloud.header.frame_id = "base_link";
      createCloud(obj_base, cloud);
      obj_base_pub_->publish(cloud);
    }
    if (map_received_) {
      sensor_msgs::msg::PointCloud2 cloud;
      cloud.header.stamp = msg->header.stamp;
      cloud.header.frame_id = "map";
      createCloud(obj_map, cloud);
      obj_map_pub_->publish(cloud);
    }
  }

  /////////////////////////////////////////////////
  // helper: vector<xyz> -> PointCloud2
  /////////////////////////////////////////////////
  void createCloud(const std::vector<std::array<float,3>> &pts,
                   sensor_msgs::msg::PointCloud2 &cloud)
  {
    cloud.height = 1;
    cloud.width  = pts.size();
    cloud.is_dense = true;
    cloud.is_bigendian = false;

    cloud.fields.clear();
    cloud.fields.resize(3);

    cloud.fields[0].name = "x";
    cloud.fields[0].offset = 0;
    cloud.fields[0].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[0].count = 1;

    cloud.fields[1].name = "y";
    cloud.fields[1].offset = 4;
    cloud.fields[1].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[1].count = 1;

    cloud.fields[2].name = "z";
    cloud.fields[2].offset = 8;
    cloud.fields[2].datatype = sensor_msgs::msg::PointField::FLOAT32;
    cloud.fields[2].count = 1;

    cloud.point_step = 12;
    cloud.row_step   = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step * cloud.height);

    for (size_t i=0; i<pts.size(); ++i)
    {
      float *fp = reinterpret_cast<float*>(&cloud.data[i*12]);
      fp[0] = pts[i][0];
      fp[1] = pts[i][1];
      fp[2] = pts[i][2];
    }
  }

  // 멤버들
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr mask_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr yolo_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pf_sub_;

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lane_base_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lane_map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obj_base_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr obj_map_pub_;

  double fx_, fy_, cx_, cy_;
  cv::Mat mask_;
  int img_w_, img_h_;
  std::mutex mask_mutex_;

  std::vector<YOLOBox> yolo_boxes_;
  std::mutex yolo_mutex_;

  double cam_tz_;
  double cam_pitch_;
  double lane_width_;

  bool map_received_;
  std::array<std::array<double,3>,3> map_R_;
  std::array<double,3> map_t_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LanePointProjectorCpp>());
  rclcpp::shutdown();
  return 0;
}
