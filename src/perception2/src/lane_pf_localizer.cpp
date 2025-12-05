#include <memory>
#include <vector>
#include <random>
#include <cmath>
#include <array>

#include "rclcpp/rclcpp.hpp"

#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/pose_with_covariance_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "std_msgs/msg/header.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include "tf2/LinearMath/Quaternion.h"

using std::placeholders::_1;

namespace
{
// ---------------------- 유틸 ------------------------
double normalize_angle(double a)
{
  a = std::fmod(a + M_PI, 2.0 * M_PI);
  if (a < 0.0)
  {
    a += 2.0 * M_PI;
  }
  return a - M_PI;
}

double quat_to_yaw(const geometry_msgs::msg::Quaternion &q)
{
  const double x = q.x;
  const double y = q.y;
  const double z = q.z;
  const double w = q.w;

  const double siny_cosp = 2.0 * (w * z + x * y);
  const double cosy_cosp = 1.0 - 2.0 * (y * y + z * z);
  return std::atan2(siny_cosp, cosy_cosp);
}

}  // namespace

// =====================================================
//       Particle Filter Localizer (Map-matching) (C++)
// =====================================================
class LanePFLocalizerCpp : public rclcpp::Node
{
public:
  LanePFLocalizerCpp()
  : Node("lane_pf_localizer_map"),
    has_map_(false),
    particles_initialized_(false),
    rng_(std::random_device{}())
  {
    // ---------------- parameters -----------------
    int num_particles = this->declare_parameter<int>("num_particles", 300);
    std::string odom_topic =
      this->declare_parameter<std::string>("odom_topic", "/odometry/filtered");
    std::string lane_topic =
      this->declare_parameter<std::string>("lane_center_topic",
                                           "/lane_points_base");
    std::string pf_pose_topic =
      this->declare_parameter<std::string>("pf_pose_topic", "/lane_pf_pose");

    double motion_xy_noise =
      this->declare_parameter<double>("motion_xy_noise", 0.01);
    double motion_yaw_noise_deg =
      this->declare_parameter<double>("motion_yaw_noise_deg", 1.0);

    int max_lane_points =
      this->declare_parameter<int>("max_lane_points", 60);

    int occ_high =
      this->declare_parameter<int>("occ_high", 70);
    int occ_mid =
      this->declare_parameter<int>("occ_mid", 30);

    // ---------------- load parameters -------------
    N_ = num_particles;
    odom_topic_ = odom_topic;
    lane_topic_ = lane_topic;
    pf_pose_topic_ = pf_pose_topic;

    motion_xy_noise_ = motion_xy_noise;
    motion_yaw_noise_ = motion_yaw_noise_deg * M_PI / 180.0;
    max_lane_points_ = max_lane_points;
    occ_high_ = occ_high;
    occ_mid_ = occ_mid;

    particles_.resize(N_);
    weights_.assign(N_, 1.0 / static_cast<double>(N_));

    RCLCPP_INFO(
      this->get_logger(),
      "PF Localizer (C++) started: N=%d, using OccupancyGrid /map", N_);

    // ---------------- subscribers -----------------
    map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
      "/map", 10, std::bind(&LanePFLocalizerCpp::map_callback, this, _1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 50, std::bind(&LanePFLocalizerCpp::odom_callback, this, _1));

    lane_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      lane_topic_, 10, std::bind(&LanePFLocalizerCpp::lane_callback, this, _1));

    // ---------------- publishers ------------------
    pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
      pf_pose_topic_, 10);
    particles_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>(
      "/lane_pf_particles", 10);
  }

private:
  // =====================================================
  //                      MAP CALLBACK
  // =====================================================
  void map_callback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg)
  {
    map_res_ = msg->info.resolution;
    map_origin_x_ = msg->info.origin.position.x;
    map_origin_y_ = msg->info.origin.position.y;
    map_width_ = msg->info.width;
    map_height_ = msg->info.height;

    // msg->data: std::vector<int8_t> (0..100, -1)
    map_data_.assign(msg->data.begin(), msg->data.end());
    has_map_ = true;

    RCLCPP_INFO(
      this->get_logger(),
      "[map] loaded occupancy grid: %u x %u, res=%.4f",
      map_width_, map_height_, map_res_);
  }

  // map frame → grid index
  bool world_to_map_index(double x, double y, int &u, int &v) const
  {
    if (!has_map_)
    {
      return false;
    }

    u = static_cast<int>((x - map_origin_x_) / map_res_);
    v = static_cast<int>((y - map_origin_y_) / map_res_);

    if (u < 0 || v < 0 || static_cast<unsigned int>(u) >= map_width_ ||
        static_cast<unsigned int>(v) >= map_height_)
    {
      return false;
    }
    return true;
  }

  // =====================================================
  //                      ODOM CALLBACK
  // =====================================================
  void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!particles_initialized_)
    {
      init_particles_from_odom(*msg);
      last_odom_ = *msg;
      return;
    }

    // 첫 init 이후이지만 last_odom_이 아직 비었을 수 있는 경우 보호
    if (!last_odom_valid_)
    {
      last_odom_ = *msg;
      last_odom_valid_ = true;
      return;
    }

    // compute delta
    double x_prev = last_odom_.pose.pose.position.x;
    double y_prev = last_odom_.pose.pose.position.y;
    double yaw_prev = quat_to_yaw(last_odom_.pose.pose.orientation);

    double x_now = msg->pose.pose.position.x;
    double y_now = msg->pose.pose.position.y;
    double yaw_now = quat_to_yaw(msg->pose.pose.orientation);

    double dx = x_now - x_prev;
    double dy = y_now - y_prev;
    double dyaw = normalize_angle(yaw_now - yaw_prev);

    last_odom_ = *msg;
    last_odom_valid_ = true;

    predict(dx, dy, dyaw);
  }

  // init particles around odom
  void init_particles_from_odom(const nav_msgs::msg::Odometry &msg)
  {
    double x0 = msg.pose.pose.position.x;
    double y0 = msg.pose.pose.position.y;
    double yaw0 = quat_to_yaw(msg.pose.pose.orientation);

    double sigma_xy = 0.05;                        // 5cm
    double sigma_yaw = 5.0 * M_PI / 180.0;         // 5 deg

    std::normal_distribution<double> dist_xy(0.0, sigma_xy);
    std::normal_distribution<double> dist_yaw(0.0, sigma_yaw);

    for (int i = 0; i < N_; ++i)
    {
      Particle p;
      p.x = x0 + dist_xy(rng_);
      p.y = y0 + dist_xy(rng_);
      p.yaw = yaw0 + dist_yaw(rng_);
      p.yaw = normalize_angle(p.yaw);
      particles_[i] = p;
      weights_[i] = 1.0 / static_cast<double>(N_);
    }

    particles_initialized_ = true;
    last_odom_valid_ = true;

    RCLCPP_INFO(
      this->get_logger(),
      "[init] init around (%.2f, %.2f, %.1f deg)",
      x0, y0, yaw0 * 180.0 / M_PI);
  }

  // prediction step
  void predict(double dx, double dy, double dyaw)
  {
    if (!particles_initialized_)
      return;

    std::normal_distribution<double> dist_xy(0.0, motion_xy_noise_);
    std::normal_distribution<double> dist_yaw(0.0, motion_yaw_noise_);

    for (int i = 0; i < N_; ++i)
    {
      particles_[i].x += dx + dist_xy(rng_);
      particles_[i].y += dy + dist_xy(rng_);
      particles_[i].yaw += dyaw + dist_yaw(rng_);
      particles_[i].yaw = normalize_angle(particles_[i].yaw);
    }
  }

  // =====================================================
  //             LANE POINTCLOUD (measurement)
  // =====================================================
  void lane_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!(particles_initialized_ && has_map_))
    {
      return;
    }

    // pointcloud → vector of (x,y)
    std::vector<std::array<double, 2>> pts;
    pts.reserve(128);

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");

    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y)
    {
      float Xb = *iter_x;
      float Yb = *iter_y;
      // Z는 사용 안함

      // NaN 처리 및 기타는 sensor_msgs iterator가 어느 정도 해줌
      // 필요하면 여기서 추가 검사 가능
      pts.push_back({static_cast<double>(Xb), static_cast<double>(Yb)});
    }

    if (pts.empty())
    {
      return;
    }

    // downsampling
    int M = static_cast<int>(pts.size());
    if (M > max_lane_points_)
    {
      std::vector<std::array<double, 2>> downsampled;
      downsampled.reserve(max_lane_points_);

      double step = static_cast<double>(M - 1) /
                    static_cast<double>(max_lane_points_ - 1);
      for (int i = 0; i < max_lane_points_; ++i)
      {
        int idx = static_cast<int>(std::round(i * step));
        if (idx < 0)
          idx = 0;
        if (idx >= M)
          idx = M - 1;
        downsampled.push_back(pts[idx]);
      }
      pts.swap(downsampled);
      M = max_lane_points_;
    }

    std::vector<double> new_weights(N_, 0.0);

    // for each particle: evaluate likelihood
    for (int i = 0; i < N_; ++i)
    {
      const double X_i = particles_[i].x;
      const double Y_i = particles_[i].y;
      const double yaw_i = particles_[i].yaw;

      const double cos_y = std::cos(yaw_i);
      const double sin_y = std::sin(yaw_i);

      double score_sum = 0.0;
      int valid = 0;

      for (int j = 0; j < M; ++j)
      {
        const double xb = pts[j][0];
        const double yb = pts[j][1];

        // lane points (base_link) → map frame
        const double Xs_m = X_i + cos_y * xb - sin_y * yb;
        const double Ys_m = Y_i + sin_y * xb + cos_y * yb;

        int u = 0, v = 0;
        if (!world_to_map_index(Xs_m, Ys_m, u, v))
        {
          continue;
        }

        // map_data_는 row-major: index = v*width + u
        const int idx = v * static_cast<int>(map_width_) + u;
        if (idx < 0 ||
            idx >= static_cast<int>(map_data_.size()))
        {
          continue;
        }

        const int occ = static_cast<int>(map_data_[idx]);  // -1 or 0..100

        double s = 0.0;
        if (occ == -1)
        {
          s = 0.3;
        }
        else if (occ >= occ_high_)
        {
          s = 1.0;
        }
        else if (occ >= occ_mid_)
        {
          s = 0.5;
        }
        else
        {
          s = 0.1;
        }

        score_sum += s;
        valid += 1;
      }

      if (valid == 0)
      {
        new_weights[i] = weights_[i];
      }
      else
      {
        double S_i = score_sum / static_cast<double>(valid);
        new_weights[i] = weights_[i] * S_i;
      }
    }

    // normalize
    double sw = 0.0;
    for (double w : new_weights)
    {
      sw += w;
    }

    if (sw < 1e-12)
    {
      RCLCPP_WARN(this->get_logger(), "all weights ~0, reset");
      double uniform_w = 1.0 / static_cast<double>(N_);
      for (int i = 0; i < N_; ++i)
      {
        weights_[i] = uniform_w;
      }
    }
    else
    {
      for (int i = 0; i < N_; ++i)
      {
        weights_[i] = new_weights[i] / sw;
      }
    }

    // resample
    resample_if_needed();

    // publish final pose
    publish_estimated_pose(msg->header.stamp);
  }

  // =====================================================
  //                 Resample & Helpers
  // =====================================================
  void resample_if_needed()
  {
    double sum_w2 = 0.0;
    for (double w : weights_)
    {
      sum_w2 += w * w;
    }
    if (sum_w2 <= 0.0)
      return;

    double neff = 1.0 / sum_w2;
    if (neff < static_cast<double>(N_) * 0.5)
    {
      systematic_resample();
    }
  }

  void systematic_resample()
  {
    const int N = N_;
    std::vector<Particle> new_particles(N);
    std::vector<double> new_weights(N, 1.0 / static_cast<double>(N));

    std::uniform_real_distribution<double> dist_u(0.0, 1.0);
    double start = dist_u(rng_) / static_cast<double>(N);

    std::vector<double> cumsum(N);
    cumsum[0] = weights_[0];
    for (int i = 1; i < N; ++i)
    {
      cumsum[i] = cumsum[i - 1] + weights_[i];
    }

    int i = 0;
    int j = 0;
    while (i < N)
    {
      double u = start + static_cast<double>(i) / static_cast<double>(N);
      while (j < N && u > cumsum[j])
      {
        ++j;
      }
      if (j == N)
      {
        j = N - 1;
      }
      new_particles[i] = particles_[j];
      ++i;
    }

    particles_.swap(new_particles);
    weights_.swap(new_weights);
  }

  // =====================================================
  //                 Publish Final Pose
  // =====================================================
  void publish_estimated_pose(const rclcpp::Time &stamp)
  {
    if (!particles_initialized_)
      return;

    // weight 평균
    double x_mean = 0.0;
    double y_mean = 0.0;
    double c_mean = 0.0;
    double s_mean = 0.0;

    for (int i = 0; i < N_; ++i)
    {
      const double w = weights_[i];
      x_mean += w * particles_[i].x;
      y_mean += w * particles_[i].y;
      c_mean += w * std::cos(particles_[i].yaw);
      s_mean += w * std::sin(particles_[i].yaw);
    }

    double yaw_mean = std::atan2(s_mean, c_mean);

    geometry_msgs::msg::PoseWithCovarianceStamped msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = "map";

    msg.pose.pose.position.x = x_mean;
    msg.pose.pose.position.y = y_mean;
    msg.pose.pose.position.z = 0.0;

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw_mean);
    msg.pose.pose.orientation.x = q.x();
    msg.pose.pose.orientation.y = q.y();
    msg.pose.pose.orientation.z = q.z();
    msg.pose.pose.orientation.w = q.w();

    // simple covariance
    std::array<double, 36> cov{};
    for (double &v : cov)
      v = 0.0;
    cov[0] = 0.05;                                   // x
    cov[7] = 0.05;                                   // y
    cov[35] = std::pow(5.0 * M_PI / 180.0, 2);       // yaw
    msg.pose.covariance = cov;
    pose_pub_->publish(msg);

    // publish particles (debug)
    geometry_msgs::msg::PoseArray pa;
    pa.header = msg.header;

    pa.poses.reserve(N_);
    for (int i = 0; i < N_; ++i)
    {
      geometry_msgs::msg::Pose pi;
      pi.position.x = particles_[i].x;
      pi.position.y = particles_[i].y;
      pi.position.z = 0.0;

      tf2::Quaternion qi;
      qi.setRPY(0.0, 0.0, particles_[i].yaw);
      pi.orientation.x = qi.x();
      pi.orientation.y = qi.y();
      pi.orientation.z = qi.z();
      pi.orientation.w = qi.w();

      pa.poses.push_back(pi);
    }

    particles_pub_->publish(pa);
  }

private:
  struct Particle
  {
    double x;
    double y;
    double yaw;
  };

  // 파라미터/상수
  int N_;
  std::string odom_topic_;
  std::string lane_topic_;
  std::string pf_pose_topic_;

  double motion_xy_noise_;
  double motion_yaw_noise_;
  int max_lane_points_;
  int occ_high_;
  int occ_mid_;

  // map
  bool has_map_;
  double map_res_;
  double map_origin_x_;
  double map_origin_y_;
  unsigned int map_width_;
  unsigned int map_height_;
  std::vector<int8_t> map_data_;

  // particles
  bool particles_initialized_;
  std::vector<Particle> particles_;
  std::vector<double> weights_;
  nav_msgs::msg::Odometry last_odom_;
  bool last_odom_valid_{false};

  // ROS
  rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lane_sub_;

  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr particles_pub_;

  // random
  std::mt19937 rng_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LanePFLocalizerCpp>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
