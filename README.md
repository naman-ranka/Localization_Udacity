# Scan Matching Localization


This is the project for the third course in the  [Udacity Self-Driving Car Engineer Nanodegree Program](https://www.udacity.com/course/c-plus-plus-nanodegree--nd213) : Scan Matching Localization

**Summary**:

The objective of this project was to achieve accurate localization of a simulated car in a distance pose error of 1.2m, while traveling a minimum of 170m from its starting position. The car was equipped with a lidar that generated regular scans, and a point cloud map (map.pcd) was provided. The project utilized the Iterative Closest Point (ICP) and Normal Distribution Transform (NDT) algorithms for matching lidar scans to the point cloud map. The implementation of both algorithms was developed from scratch. Ultimately, the project achieved the goal of accurate localization of the car in simulation while adhering to the distance pose error constraint.

![this is a image](/img/png_to_gif.gif)


## Code Snippets

### ICP Algoritm(From Scratch):

```
Eigen::Matrix4d ICP1( PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose, int iterations,pcl::KdTreeFLANN<PointT> &kdtree){

  	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
  	Eigen::Matrix4d final_transformation_matrix = Eigen::Matrix4d::Identity();
  	Eigen::Matrix4d initTransform = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll, startingPose.position.x, startingPose.position.y, startingPose.position.z);

  	PointCloudT::Ptr source_trans(new PointCloudT);
  	PointCloudT::Ptr source_trans1(new PointCloudT);
  
  	pcl::transformPointCloud(*source,*source_trans,initTransform);
  	pcl::transformPointCloud(*source,*source_trans1,initTransform);
  
  	for(int i =0;i<iterations;i++){
      		pcl::transformPointCloud(*source_trans1,*source_trans,final_transformation_matrix);
      		//Making associations between source and target point clouds
            vector<int> associations;
            for(PointT p:source_trans->points){
                vector<int> pointIdxRadiusSearch;
                vector<float> pointRadiusSquaredDistance;
                if(kdtree.radiusSearch(p,1.0,pointIdxRadiusSearch,pointRadiusSquaredDistance)>0){
                    associations.push_back(pointIdxRadiusSearch[0]);            
                }
                else{
                    associations.push_back(-1);
                }
            }
			//Pairing the associated points
            vector<Pair> pairs = PairPoints(associations,target,source_trans);
            int index = 0;
            Eigen::MatrixXd P = Eigen::MatrixXd::Zero(3,1);
            Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(3,1);
            //Calculating centroid for target(P) and source(Q)
      		for(Pair p:pairs){
                P(0,0) += p.p1.x;
                Q(0,0) += p.p2.x;
                P(1,0) += p.p1.y;
                Q(1,0) += p.p2.y;
                P(2,0) += p.p1.z;
                Q(2,0) += p.p2.z;
                index++;        
              }
            P(0,0) = P(0,0)/index;
            P(1,0) = P(1,0)/index;
            P(2,0) = P(2,0)/index;
            Q(0,0) = Q(0,0)/index;	
            Q(1,0) = Q(1,0)/index;
            Q(2,0) = Q(2,0)/index;

            Eigen::MatrixXd X(3,index);
            Eigen::MatrixXd Y(3,index);

            for(int i=0;i<index;i++){
                X(0,i) = pairs[i].p1.x - P(0,0);
                X(1,i) = pairs[i].p1.y - P(1,0);
                X(2,i) = pairs[i].p1.z - P(2,0);
                Y(0,i) = pairs[i].p2.x - Q(0,0);
                Y(1,i) = pairs[i].p2.y - Q(1,0);
                Y(2,i) = pairs[i].p2.z - Q(2,0);
            }
			//Perform Singular Value Decomposition on S
            Eigen::MatrixXd S = X * Y.transpose();
            JacobiSVD<MatrixXd> svd(S, ComputeFullV | ComputeFullU);
            Eigen::MatrixXd I ;
            I.setIdentity(svd.matrixV().cols(), svd.matrixU().cols());
            I(svd.matrixV().cols()-1,svd.matrixU().cols()-1) = (svd.matrixV() * svd.matrixU().transpose()).determinant();            
            Eigen::MatrixXd R = svd.matrixV() * I * svd.matrixU().transpose();            
            Eigen::MatrixXd t = Q - R*P;
            
            transformation_matrix(0,0) = R(0,0);
            transformation_matrix(0,1) = R(0,1);
            transformation_matrix(0,2) = R(0,2);
            transformation_matrix(1,0) = R(1,0);
            transformation_matrix(1,1) = R(1,1);
            transformation_matrix(1,2) = R(1,2);
            transformation_matrix(2,0) = R(2,0);
            transformation_matrix(2,1) = R(2,1);
            transformation_matrix(2,2) = R(2,2);
            transformation_matrix(0,3) = t(0,0);
            transformation_matrix(1,3) = t(1,0);
            transformation_matrix(2,3) = t(2,0);
      
            final_transformation_matrix *= transformation_matrix;           
    }

  	return final_transformation_matrix*initTransform;
}
```


### NDT Algorithm(PCL library):

```
//NDT algorithm from pcl libarary pcl/registration/ndt.h
Eigen::Matrix4d NDT(pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt, PointCloudT::Ptr source, Pose startingPose, int iterations){

  	Eigen::Matrix4f init_guess = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll, startingPose.position.x, startingPose.position.y, startingPose.position.z).cast<float>();
	
  	ndt.setMaximumIterations (iterations);
	ndt.setInputSource (source);
  	
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ndt (new pcl::PointCloud<pcl::PointXYZ>);
  	ndt.align (*cloud_ndt, init_guess);

	Eigen::Matrix4d transformation_matrix = ndt.getFinalTransformation ().cast<double>();

	return transformation_matrix;
}

```



### ICP Algorithm(PCL library):

```
Eigen::Matrix4d ICP(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose, int iterations){

  	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity ();
  	Eigen::Matrix4d trans = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll, startingPose.position.x, startingPose.position.y, startingPose.position.z);
    PointCloudT::Ptr source_tans(new PointCloudT);
  	pcl::transformPointCloud (*source, *source_tans,trans);
  	pcl::IterativeClosestPoint<PointT, PointT> icp;
	icp.setInputSource(source_tans);
	icp.setInputTarget(target);
  	icp.setMaximumIterations (iterations);
  	icp.setMaxCorrespondenceDistance (2.0);
    PointCloudT::Ptr Final(new PointCloudT);
    icp.align(*Final);
  	if(icp.hasConverged()){
    	transformation_matrix = icp.getFinalTransformation().cast<double>();
      	transformation_matrix = transformation_matrix*trans;
      	return transformation_matrix;
    }
  	return transformation_matrix*trans;
}

```


## Udacity Project Review:


![this is a image](/img/LocalizationReview.png)
