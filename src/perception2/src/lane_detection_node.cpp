#include <memory>
#include <vector>
#include <cmath>
#include <array>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/point_field.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "std_msgs/msg/header.hpp"

#include "cv_bridge/cv_bridge.h"
#include <opencv2/opencv.hpp>

#include "sensor_msgs/point_cloud2_iterator.hpp"

class LanePointProjectorCpp : public rclcpp::Node
{
public:
  LanePointProjectorCpp()
  : Node("lane_point_projector_cpp"),
    fx_(0.0), fy_(0.0), cx_(0.0), cy_(0.0),
    cam_tx_(0.0), cam_ty_(0.0), cam_tz_(0.3),
    cam_roll_(0.0), cam_pitch_(0.436332), cam_yaw_(0.0),
    image_logged_(false), cloud_logged_(false),
    map_received_(false)
  {
    // 파라미터: lane_width
    lane_width_ = declare_parameter<double>("lane_width", 0.35);
    RCLCPP_INFO(get_logger(), "[init] lane_width = %.3f m", lane_width_);

    // 토픽 이름
    std::string color_image_topic = "/camera/color/image_raw";
    std::string caminfo_topic     = "/camera/color/camera_info";
    std::string cloud_topic       = "/camera/depth/color/points";

    // Sensor QoS (BEST_EFFORT, KEEP_LAST(10))
    rclcpp::QoS qos_sensor(10);
    qos_sensor.best_effort();

    // 구독자
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      color_image_topic, qos_sensor,
      std::bind(&LanePointProjectorCpp::imageCallback, this, std::placeholders::_1));

    caminfo_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      caminfo_topic, qos_sensor,
      std::bind(&LanePointProjectorCpp::camInfoCallback, this, std::placeholders::_1));

    cloud_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      cloud_topic, qos_sensor,
      std::bind(&LanePointProjectorCpp::cloudCallback, this, std::placeholders::_1));

    pf_sub_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
      "/lane_pf_pose", 10,
      std::bind(&LanePointProjectorCpp::pfCallback, this, std::placeholders::_1));

    // 퍼블리셔
    debug_img_pub_ = create_publisher<sensor_msgs::msg::Image>("/debug/lane_points_image", 10);

    lane_cloud_cam_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/camera/depth/lane_points_camera", 10);
    lane_cloud_base_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/lane_points_base", 10);
    center_cloud_base_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/camera/depth/lane_center_base_sdf", 10);
    center_cloud_map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/camera/depth/lane_center_map", 10);
    lane_cloud_map_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/lane_points_map", 10);

    RCLCPP_INFO(get_logger(), "LanePointProjectorCpp node started.");
  }

private:
  // ====== CameraInfo 콜백 ======
  void camInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    fx_ = msg->k[0];
    cx_ = msg->k[2];
    fy_ = msg->k[4];
    cy_ = msg->k[5];

    RCLCPP_INFO(get_logger(),
                "[caminfo_callback] fx=%.2f, fy=%.2f, cx=%.2f, cy=%.2f",
                fx_, fy_, cx_, cy_);
  }

  // ====== 쿼터니언 -> 회전행렬 ======
  std::array<std::array<double, 3>, 3>
  quatToRot(double x, double y, double z, double w)
  {
    double xx = x * x;
    double yy = y * y;
    double zz = z * z;
    double xy = x * y;
    double xz = x * z;
    double yz = y * z;
    double wx = w * x;
    double wy = w * y;
    double wz = w * z;

    std::array<std::array<double, 3>, 3> R;
    R[0][0] = 1.0 - 2.0 * (yy + zz);
    R[0][1] =       2.0 * (xy - wz);
    R[0][2] =       2.0 * (xz + wy);
    R[1][0] =       2.0 * (xy + wz);
    R[1][1] = 1.0 - 2.0 * (xx + zz);
    R[1][2] =       2.0 * (yz - wx);
    R[2][0] =       2.0 * (xz - wy);
    R[2][1] =       2.0 * (yz + wx);
    R[2][2] = 1.0 - 2.0 * (xx + yy);
    return R;
  }

  // ====== lane_pf_pose 콜백 (map -> base_link) ======
  void pfCallback(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg)
  {
    double px = msg->pose.pose.position.x;
    double py = msg->pose.pose.position.y;
    double pz = msg->pose.pose.position.z;

    double qx = msg->pose.pose.orientation.x;
    double qy = msg->pose.pose.orientation.y;
    double qz = msg->pose.pose.orientation.z;
    double qw = msg->pose.pose.orientation.w;

    map_R_ = quatToRot(qx, qy, qz, qw);
    map_t_ = {px, py, pz};
    map_received_ = true;
  }

  // ====== 이미지 콜백 (흰 차선 마스크) ======
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    }
    catch (cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    latest_image_ = cv_ptr->image;

    // HSV 변환
    cv::Mat hsv;
    cv::cvtColor(latest_image_, hsv, cv::COLOR_BGR2HSV);

    // 흰색 threshold
    cv::Scalar lower_white(0, 0, 180);
    cv::Scalar upper_white(180, 60, 255);
    cv::Mat mask;
    cv::inRange(hsv, lower_white, upper_white, mask);

    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_DILATE, kernel);

    white_mask_ = mask;

    if (!image_logged_)
    {
      RCLCPP_INFO(get_logger(), "[image_callback] First image & white mask generated.");
      image_logged_ = true;
    }
  }

  // ====== PointCloud 콜백 ======
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (fx_ == 0.0 || latest_image_.empty() || white_mask_.empty())
    {
      RCLCPP_DEBUG(get_logger(),
                   "[cloud_callback] Waiting for camera info / image / mask...");
      return;
    }

    if (!cloud_logged_)
    {
      RCLCPP_INFO(get_logger(), "[cloud_callback] First pointcloud received.");
      cloud_logged_ = true;
    }

    // ✅ 더 이상 clone 안 쓰고, 크기만 사용
    const int h = latest_image_.rows;
    const int w = latest_image_.cols;

    // ✅ 마스크는 레퍼런스로만 사용
    const cv::Mat &mask = white_mask_;

    std::vector<std::array<float, 3>> lane_points_cam;
    std::vector<std::array<float, 3>> lane_points_base;

    // 대략적인 포인트 수 예측해서 reserve (선택적 최적화)
    const int step = 5;
    const size_t approx_points =
      (static_cast<size_t>(msg->width) * msg->height) / step;
    lane_points_cam.reserve(approx_points);
    lane_points_base.reserve(approx_points);

    // PointCloud2 iterator
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

    size_t i = 0;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++i)
    {
      if (i % step != 0)
        continue;

      float Xc = *iter_x;
      float Yc = *iter_y;
      float Zc = *iter_z;

      // if (Zc <= 0.0f)
        // continue;
// 
      // 이미지 좌표 투영
      float u = static_cast<float>(fx_) * (Xc / Zc) + static_cast<float>(cx_);
      float v = static_cast<float>(fy_) * (Yc / Zc) + static_cast<float>(cy_);

      int u_i = static_cast<int>(std::round(u));
      int v_i = static_cast<int>(std::round(v));

      if (u_i < 0 || u_i >= w || v_i < 0 || v_i >= h)
        continue;

      // 흰색 마스크 위치만 사용
      if (mask.at<uint8_t>(v_i, u_i) == 0)
        continue;

      // 1) optical frame 기준 포인트
      lane_points_cam.push_back({Xc, Yc, Zc});

      // 2) optical -> camera_link 변환 (REP-103)
      // optical: X-right, Y-down, Z-forward
      // camera_link: X-forward, Y-left, Z-up
      float Xl = Zc;
      float Yl = -Xc;
      float Zl = -Yc;

      // 3) camera_link -> base_link (pitch around Y축)
      double theta = cam_pitch_;
      float Xb = static_cast<float>(std::cos(theta) * Xl + std::sin(theta) * Zl);
      float Yb = Yl;
      float Zb = static_cast<float>(-std::sin(theta) * Xl + std::cos(theta) * Zl);

      // 카메라 높이 적용
      Zb += static_cast<float>(cam_tz_);

      // ===== 범위 필터 =====
      const float Xb_min    = 0.15f;
      const float Xb_max    = 1.0f;
      // const float half_lane = 0.4f * 0.5f;   // 트랙 폭의 절반
      const float half_lane = static_cast<float>(lane_width_ * 0.5); // 파라미터 기반
      const float margin_y  = 0.10f;
      // const float Z_eps     = 0.1f;

      if (Xb < Xb_min || Xb > Xb_max)
        continue;

      if (std::fabs(Yb) > (half_lane + margin_y))
        continue;

      // if (std::fabs(Zb) > Z_eps)
      //   continue;

      lane_points_base.push_back({Xb, Yb, Zb});
    }

    // ========== camera 프레임 기준 lane cloud publish ==========
    if (!lane_points_cam.empty())
    {
      sensor_msgs::msg::PointCloud2 cloud_cam;
      cloud_cam.header = msg->header;  // frame_id = camera_depth_optical_frame

      createPointCloud2FromPoints(lane_points_cam, cloud_cam);
      lane_cloud_cam_pub_->publish(cloud_cam);
    }

    // ========== base_link 기준 lane cloud publish ==========
    if (!lane_points_base.empty())
    {
      sensor_msgs::msg::PointCloud2 cloud_base;
      cloud_base.header.stamp = msg->header.stamp;
      cloud_base.header.frame_id = "base_link";

      createPointCloud2FromPoints(lane_points_base, cloud_base);
      lane_cloud_base_pub_->publish(cloud_base);

      // map 기준 lane 포인트 (lane_pf_pose 적용)
      if (map_received_)
      {
        std::vector<std::array<float, 3>> lane_points_map;
        lane_points_map.reserve(lane_points_base.size());

        for (const auto &p_base : lane_points_base)
        {
          double Xb = p_base[0];
          double Yb = p_base[1];
          double Zb = p_base[2];

          double Xmap =
            map_R_[0][0] * Xb + map_R_[0][1] * Yb + map_R_[0][2] * Zb + map_t_[0];
          double Ymap =
            map_R_[1][0] * Xb + map_R_[1][1] * Yb + map_R_[1][2] * Zb + map_t_[1];
          double Zmap =
            map_R_[2][0] * Xb + map_R_[2][1] * Yb + map_R_[2][2] * Zb + map_t_[2];

          lane_points_map.push_back(
            {static_cast<float>(Xmap),
             static_cast<float>(Ymap),
             static_cast<float>(Zmap)});
        }

        sensor_msgs::msg::PointCloud2 cloud_map;
        cloud_map.header.stamp = msg->header.stamp;
        cloud_map.header.frame_id = "map";

        createPointCloud2FromPoints(lane_points_map, cloud_map);
        lane_cloud_map_pub_->publish(cloud_map);
      }
    }

    // ✅ 디버그 이미지 관련 코드는 전부 제거 (clone / circle / imshow / publish)
  }

  // ====== helper: std::vector<xyz> -> PointCloud2 ======
  void createPointCloud2FromPoints(
    const std::vector<std::array<float, 3>> &points,
    sensor_msgs::msg::PointCloud2 &cloud)
  {
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(points.size());
    cloud.is_bigendian = false;
    cloud.is_dense = true;

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

    cloud.point_step = 12;  // 3 * 4 bytes
    cloud.row_step = cloud.point_step * cloud.width;
    cloud.data.resize(cloud.row_step * cloud.height);

    // 데이터 채우기
    for (size_t i = 0; i < points.size(); ++i)
    {
      uint8_t *ptr = &cloud.data[i * cloud.point_step];
      float *fp = reinterpret_cast<float *>(ptr);
      fp[0] = points[i][0];
      fp[1] = points[i][1];
      fp[2] = points[i][2];
    }
  }

private:
  // 구독/퍼블리셔
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr caminfo_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pf_sub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_img_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lane_cloud_cam_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lane_cloud_base_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_cloud_base_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_cloud_map_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr lane_cloud_map_pub_;

  // 카메라 intrinsics
  double fx_, fy_, cx_, cy_;

  // 카메라→base_link(고정) 변환 (SDF 기반)
  double cam_tx_, cam_ty_, cam_tz_;
  double cam_roll_, cam_pitch_, cam_yaw_;

  // 최신 이미지 / 흰색 마스크
  cv::Mat latest_image_;
  cv::Mat white_mask_;

  // PF 기반 map->base_link 변환
  std::array<std::array<double, 3>, 3> map_R_;
  std::array<double, 3> map_t_;
  bool map_received_;

  // 기타
  bool image_logged_;
  bool cloud_logged_;
  double lane_width_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LanePointProjectorCpp>();
  rclcpp::spin(node);
  rclcpp::shutdown();

  cv::destroyAllWindows();
  return 0;
}
