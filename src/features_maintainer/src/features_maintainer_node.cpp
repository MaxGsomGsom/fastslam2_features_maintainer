#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/PointCloud2.h>
#include <laser_geometry/laser_geometry.h>
#include <pointmatcher_ros/point_cloud.h>
#include <pointmatcher/PointMatcher.h>
#include <pointmatcher_ros/transform.h>
#include <tf/transform_listener.h>


using namespace PointMatcherSupport;
using namespace std;

typedef PointMatcher<float> PM;
typedef PM::DataPoints DP;

//Publishers
ros::Publisher rawCloudPublisher;
ros::Publisher inliersCloudPublisher;
ros::Publisher mapCloudPublisher;

//Listener for pose to map transformation
tf::TransformListener* tfListener;

//Params
double cutOff = 3.9;
double rateHZ = 2;
string mapFrame = "map";
string poseFrame = "base_scan";

DP mapPoints;
PM::ICP icp;
laser_geometry::LaserProjection projector;


void scan_callback(const sensor_msgs::LaserScan& msg);
void maintain_features(DP& readPoints);


int main(int argc, char** argv)
{
    ros::init(argc, argv, "features_maintainer_node");

    // Create the default ICP algorithm
    icp.setDefault();

    //Create node and topics
    ros::NodeHandle node("~");
    tfListener = new tf::TransformListener();
    ros::Subscriber scanSubscriber = node.subscribe("/scan", 1, scan_callback);
    rawCloudPublisher = node.advertise<sensor_msgs::PointCloud2>("/raw_cloud", 10);
    inliersCloudPublisher = node.advertise<sensor_msgs::PointCloud2>("/matched_cloud", 10);
    mapCloudPublisher = node.advertise<sensor_msgs::PointCloud2>("/map_cloud", 10);

    ros::Rate rate(rateHZ);
    while (ros::ok()) {        
        ros::spinOnce();
        rate.sleep();
    }

    return 0;
};


void scan_callback(const sensor_msgs::LaserScan& msg)
{
    //===== Get and check transformation params =====

    PM::TransformationParameters transMatrix;
    try {
        //Get transformation of new points cloud to map
        transMatrix = PointMatcher_ros::transformListenerToEigenMatrix<float>(*tfListener, mapFrame, poseFrame, ros::Time(0));
    }
    catch (tf2::TransformException& ex)
    {
        ROS_WARN("%s", ex.what());
        return;
    }

    PM::Transformation* rigidTrans = PM::get().REG(Transformation).create("RigidTransformation");

    if (!rigidTrans->checkParameters(transMatrix)) {
        std::cout << "WARNING: T does not represent a valid rigid transformation\nProjecting onto an orthogonal basis"<< std::endl;
        //T = rigidTrans->correctParameters(transMatrix);
        return;
    }

     //===== Transformation =====

    //Convert scan to ROS points cloud
    sensor_msgs::PointCloud2 cloud;
    projector.projectLaser(msg, cloud, cutOff);
    rawCloudPublisher.publish(cloud);

    //Convert ROS points cloud to libpointmatcher cloud
    DP readPoints = PointMatcher_ros::rosMsgToPointMatcherCloud<float>(cloud);

    // Compute the transformation
    PM::DataPoints readTransformedPoints =  rigidTrans->compute(readPoints, transMatrix);

    //Main function
    maintain_features(readTransformedPoints);

}

void maintain_features(DP& readPoints) {

    // ===== Init =====

    //Create map if empty
    if (mapPoints.features.cols() == 0) {
        mapPoints = DP(readPoints);
        return;
    }

    // ===== Match points with ICP =====

    // Compute the transformation to express readPoints in mapPoints
    PM::TransformationParameters T = icp(readPoints, mapPoints);

    // Transform readPoints to express it in mapPoints
    DP readPointsTransformed(readPoints);
    icp.transformations.apply(readPointsTransformed, T);

    // ===== Find matches =====

    DP read = readPointsTransformed;
    DP ref = mapPoints;

    const int readPtsCount = read.features.cols();
    const int refPtsCount = ref.features.cols();

    // Build kd-tree
    int knnRead = 10;
    int knnRef = 1;

    // Matcher to find closest points on read scan
    PM::Matcher *matcherRead(PM::get().MatcherRegistrar.create(
        "KDTreeMatcher",
        map_list_of("knn", toParam(knnRead))));
    matcherRead->init(read);

    // Find clothest points on read
    PM::Matches readMatches(knnRead, readPtsCount);
    readMatches = matcherRead->findClosests(read);

    // For every point set maximum search distanse as square root of distanse to
    // farest point, found by matcherRead
    const PM::Matrix maxSearchDist = readMatches.dists.colwise().maxCoeff().cwiseSqrt();
    read.addDescriptor("maxSearchDist", maxSearchDist);

    // Matcher to match points on both scans with limit to max distance
    PM::Matcher *matcherReadToTarget(PM::get().MatcherRegistrar.create(
        "KDTreeVarDistMatcher",
        map_list_of("knn", toParam(knnRef))
                   ("maxDistField","maxSearchDist") // descriptor name
        ));
    matcherReadToTarget->init(ref);

    // Find matches from read to ref
    PM::Matches refMatches(knnRef, refPtsCount);
    refMatches = matcherReadToTarget->findClosests(read);

    // Add new descriptors to select inliers and outliers
    const PM::Matrix inliersRead = PM::Matrix::Zero(1, read.features.cols());
    read.addDescriptor("inliers", inliersRead);
    const PM::Matrix inliersRef = PM::Matrix::Zero(1, ref.features.cols());
    ref.addDescriptor("inliers", inliersRef);

    // Get view to edit inliers
    auto inlierRead = read.getDescriptorViewByName("inliers");
    auto inlierRef = ref.getDescriptorViewByName("inliers");

    // For every point
    for (int i = 0; i < readPtsCount; i++) {
      // For every match
      for (int k = 0; k < knnRef; k++) {

        if (refMatches.dists(k, i) != PM::Matches::InvalidDist) {
          // Set inlier descriptor to both points
          inlierRead(0, i) = 1.0;
          inlierRef(0, refMatches.ids(k, i)) = 1.0;
        }

      }
    }

    // ===== Filter matches =====

    // Filter by inlier descriptor
    PM::OutlierFilter *filterInl = PM::get().OutlierFilterRegistrar.create(
        "GenericDescriptorOutlierFilter",
        map_list_of("source", "reference")
                   ("descName", "inliers")    //descriptor to filter
                   ("useSoftThreshold", "1"));
    const PM::OutlierWeights inlierWeights = filterInl->compute(read, ref, refMatches);

    // Generate tuples (pairs) of matched points and remove ones with zero weight
    const PM::ErrorMinimizer::ErrorElements matchedPoints(read, ref, inlierWeights, refMatches);


    // ===== Filter outliers =====

    // Copy outliers to new cloud
    DP outliersRead = read.createSimilarEmpty();
    int count = 0;
    for (int i = 0; i < readPtsCount; i++) {
        if (inlierRead(0,i)==0) {
            outliersRead.features.col(count) = read.features.col(i);
            outliersRead.descriptors.col(count) = read.descriptors.col(i);
            count++;
        }
    }
    outliersRead.conservativeResize(count);

    // Copy outliers to new cloud
    DP outliersRef = ref.createSimilarEmpty();
    count = 0;
    for (int i = 0; i < refPtsCount; i++) {
        if (inlierRef(0,i)==0) {
            outliersRef.features.col(count) = ref.features.col(i);
            outliersRef.descriptors.col(count) = ref.descriptors.col(i);
            count++;
        }
    }
    outliersRef.conservativeResize(count);


    // ===== Publish to ROS =====

    sensor_msgs::PointCloud2 inliersCloud = PointMatcher_ros::pointMatcherCloudToRosMsg<float>(matchedPoints.reading, mapFrame, ros::Time(0));
    sensor_msgs::PointCloud2 mapCloud = PointMatcher_ros::pointMatcherCloudToRosMsg<float>(mapPoints, mapFrame, ros::Time(0));
    inliersCloudPublisher.publish(inliersCloud);
    mapCloudPublisher.publish(mapCloud);

    // ===== Debug =====

    cout << "points=" << read.getNbPoints() << ", inliers=" << matchedPoints.reading.getNbPoints()
         << ", scan_outliers=" << outliersRead.getNbPoints() << ", map_outliers=" << outliersRef.getNbPoints() << endl;

    /*read.save("read.vtk");
    ref.save("ref.vtk");
    matchedPoints.reading.save("matched_read.vtk");
    outliersRead.save("outl_read.vtk");
    outliersRef.save("outl_ref.vtk");*/

}
