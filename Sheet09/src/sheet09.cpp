#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

cv::Mat flow_2_RGB( const cv::Mat &inpFlow, const float &max_size );
void read_FLO_file( std::string filePATH, cv::Mat &flowGroundTruth, float &maxxV );

/*
 * Average Angular Error
 */
float averageAngularError(const cv::Mat& estimatedFlow, const cv::Mat& flowGroundTruth) {

    double aae = 0;
    for (int i = 0; i < estimatedFlow.rows; i++) {
        for (int j = 0; j < estimatedFlow.cols; j++) {
            double numerator = flowGroundTruth.at<cv::Point2f>(i,j).dot(estimatedFlow.at<cv::Point2f>(i,j)) + 1.0;
            double denomPt1  = flowGroundTruth.at<cv::Point2f>(i,j).dot(flowGroundTruth.at<cv::Point2f>(i,j)) + 1.0;
            double denomPt2  = estimatedFlow.at<cv::Point2f>(i,j).dot(estimatedFlow.at<cv::Point2f>(i,j)) + 1.0;
            aae += acos( numerator / sqrt(denomPt1 * denomPt2) );
        }
    }

    aae /= estimatedFlow.rows * estimatedFlow.cols;

    return aae * 180 / M_PI; // return in degree
}

/*
 * largest absolute magnitude of flow field
 */
float maxFlow(const cv::Mat& flow) {
    double max = -99999999;
    for (int i = 0; i < flow.rows; i++) {
        for (int j = 0; j < flow.cols; j++) {
            double magnitude = cv::norm(flow.at<cv::Point2f>(i,j));
            if (magnitude > max)
                max = magnitude;
        }
    }
    return max;
}

/*
 * Lucas-Kanade
 */
void lucasKanade(const cv::Mat& IxIx, const cv::Mat& IyIy, const cv::Mat& IxIy, const cv::Mat& IxIt, const cv::Mat& IyIt, cv::Mat& flow, int windowDim) {

    cv::Size size = cv::Size(IxIx.cols,IxIx.rows);

    cv::Mat IxIx2 = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat IyIy2 = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat IxIy2 = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat IxIt2 = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat IyIt2 = cv::Mat::zeros( size, CV_32FC1 );

    // apply window function
    cv::boxFilter( IxIx, IxIx2, -1, cv::Size(windowDim, windowDim), cv::Point(-1,-1), false, cv::BORDER_DEFAULT ); // Point(-1,-1) means that the anchor is at the kernel center.
    cv::boxFilter( IyIy, IyIy2, -1, cv::Size(windowDim, windowDim), cv::Point(-1,-1), false, cv::BORDER_DEFAULT );
    cv::boxFilter( IxIy, IxIy2, -1, cv::Size(windowDim, windowDim), cv::Point(-1,-1), false, cv::BORDER_DEFAULT );
    cv::boxFilter( IxIt, IxIt2, -1, cv::Size(windowDim, windowDim), cv::Point(-1,-1), false, cv::BORDER_DEFAULT );
    cv::boxFilter( IyIt, IyIt2, -1, cv::Size(windowDim, windowDim), cv::Point(-1,-1), false, cv::BORDER_DEFAULT );

    // solve linear system for each pixel
    for (int i = 0; i < flow.rows; i++) {
        for (int j = 0; j < flow.cols; j++) {

            cv::Mat secondMoment = cv::Mat::zeros( 2,2, CV_32FC1 );
            secondMoment.at<float>(0,0) = IxIx2.at<float>(i,j);
            secondMoment.at<float>(0,1) = IxIy2.at<float>(i,j);
            secondMoment.at<float>(1,0) = IxIy2.at<float>(i,j);
            secondMoment.at<float>(1,1) = IyIy2.at<float>(i,j);

            cv::Mat b = cv::Mat::zeros( 2,1, CV_32FC1 );
            b.at<float>(0,0) = -IxIt2.at<float>(i,j);
            b.at<float>(1,0) = -IyIt2.at<float>(i,j);

            // cv::DECOMP_SVD -> SVD decomposition. If the matrix is singular or even non-square, the pseudo inverse is computed
            cv::Mat flowVect = secondMoment.inv(cv::DECOMP_SVD) * b;

            flow.at<cv::Point2f>(i,j).x = flowVect.at<float>(0,0);
            flow.at<cv::Point2f>(i,j).y = flowVect.at<float>(1,0);
        }
    }
}

/*
 * Horn-Schunck
 */
void hornSchunck(const cv::Mat& IxIx, const cv::Mat& IyIy, const cv::Mat& IxIy, const cv::Mat& IxIt, const cv::Mat& IyIt, cv::Mat& flow) {
    // matrix definitions
    cv::Mat oldU = cv::Mat::zeros( flow.size(), CV_32FC1 );
    cv::Mat oldV = cv::Mat::zeros( flow.size(), CV_32FC1 );
    cv::Mat newU = cv::Mat::zeros( flow.size(), CV_32FC1 );
    cv::Mat newV = cv::Mat::zeros( flow.size(), CV_32FC1 );
    cv::Mat laplaceU = cv::Mat::zeros( flow.size(), CV_32FC1 );
    cv::Mat laplaceV = cv::Mat::zeros( flow.size(), CV_32FC1 );
    cv::Mat tmp1 = cv::Mat::zeros( flow.size(), CV_32FC1 );
    cv::Mat tmp2 = cv::Mat::zeros( flow.size(), CV_32FC1 );

    cv::Mat alpha = cv::Mat::ones( flow.size(), CV_32FC1 );
    alpha = alpha + IxIx + IyIy;

    for (int k = 0; k < 1000; k++) {
        // apply laplace filter
        cv::Laplacian(oldU, laplaceU, -1, 1, 0.25, 0, cv::BORDER_DEFAULT);
        cv::Laplacian(oldV, laplaceV, -1, 1, 0.25, 0, cv::BORDER_DEFAULT);
        laplaceU = laplaceU + oldU;
        laplaceV = laplaceV + oldV;
    
        // update formula for U
        cv::multiply(IxIx, laplaceU, tmp1);
        cv::multiply(IxIy, laplaceV, tmp2);
        newU = tmp1 + tmp2 + IxIt;
        cv::divide(newU, alpha, newU);
        newU = laplaceU - newU;
        // update formula for V
        cv::multiply(IxIy, laplaceU, tmp1);
        cv::multiply(IyIy, laplaceV, tmp2);
        newV = tmp1 + tmp2 + IyIt;
        cv::divide(newV, alpha, newV);
        newV = laplaceV - newV;
    
        // compare change of norm
        tmp1 = newU - oldU;
        tmp2 = newV - oldV;
        double diff = cv::norm(tmp1, cv::NORM_L1) + cv::norm(tmp2, cv::NORM_L1);
        if (diff < 0.002) k = 1000;
    
        newU.copyTo(oldU);
        newV.copyTo(oldV);
    }
    
    // store result in flow
    for (int i = 0; i < flow.rows; i++) {
        for (int j = 0; j < flow.cols; j++) {
            flow.at<cv::Point2f>(i,j).x = newU.at<float>(i,j);
            flow.at<cv::Point2f>(i,j).y = newV.at<float>(i,j);
        }
    }
}

int main(int argc, char *argv[])
{

    int window_DIM = 15;

    std::string fileNameFlow  = "./data/groundTruthOF.flo";
    std::string fileNameImg1 = "./data/frame1.png";
    std::string fileNameImg2 = "./data/frame2.png";

    //////////////////////////////////////////////////////////
    cv::Mat imgRGB1 = cv::imread(fileNameImg1);
    cv::Mat imgRGB2 = cv::imread(fileNameImg2);
    cv::Mat flowGroundTruth;
    float maxFlowGroundTruth;
    read_FLO_file( fileNameFlow, flowGroundTruth, maxFlowGroundTruth );
    //////////////////////////////////////////////////////////
    cv::Mat              imgGRAY1,imgGRAY1_fl;
    cv::Mat              imgGRAY2,imgGRAY2_fl;
    cv::cvtColor(imgRGB1,imgGRAY1,CV_BGR2GRAY);
    cv::cvtColor(imgRGB2,imgGRAY2,CV_BGR2GRAY);
    imgGRAY1.convertTo(imgGRAY1_fl,CV_32FC1);
    imgGRAY2.convertTo(imgGRAY2_fl,CV_32FC1);
    //////////////////////////////////////////////////////////

    cv::Size size = cv::Size(imgGRAY1.cols,imgGRAY1.rows);

    cv::Mat Ix   = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat Iy   = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat IxIx = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat IyIy = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat IxIy = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat It   = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat IxIt = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat IyIt = cv::Mat::zeros( size, CV_32FC1 );
    cv::Mat flow = cv::Mat::zeros( size, CV_32FC2 );

    //////////////////////////////////////////////////////////////////////
    ////  Prepare Derivatives  ///////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////
    cv::Sobel( imgGRAY1, Ix, CV_32FC1, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel( imgGRAY1, Iy, CV_32FC1, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);

    cv::multiply(Ix,Ix,IxIx);
    cv::multiply(Iy,Iy,IyIy);
    cv::multiply(Ix,Iy,IxIy);

    It = (imgGRAY2_fl - imgGRAY1_fl);

    cv::multiply(Ix,It,IxIt);
    cv::multiply(Iy,It,IyIt);

    //////////////////////////////////////////////////////////////////////
    ////  Apply Lucas-Kanade  ////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    std::cout << "Applying Lucas-Kanade..." << std::endl;
    lucasKanade(IxIx, IyIy, IxIy, IxIt, IyIt, flow, window_DIM);
    std::cout << "Average Angular Error = " << averageAngularError(flow, flowGroundTruth)  << "\n" << std::endl;

    // display result
    cv::Mat flowGroundTruth_RGB = flow_2_RGB( flowGroundTruth, MAX(maxFlowGroundTruth, maxFlow(flow) ) );
    cv::Mat flow_RGB = flow_2_RGB( flow, MAX(maxFlowGroundTruth, maxFlow(flow) ) );

    cv::imshow("lucas_kanade_flow_rgb", flow_RGB);
    cv::imshow("flow_ground_truth_rgb", flowGroundTruth_RGB);

    std::cout << "Press any key to continue... \n" << std::endl;
    cv::waitKey(0);

    //////////////////////////////////////////////////////////////////////
    ////  Apply Horn-Schunck  ////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////

    std::cout << "Applying Horn-Schunck..." << std::endl;
    hornSchunck(IxIx, IyIy, IxIy, IxIt, IyIt, flow);
    std::cout << "Average Angular Error = " << averageAngularError(flow, flowGroundTruth)  << "\n" << std::endl;

    // display result
    flow_RGB = flow_2_RGB( flow, MAX(maxFlowGroundTruth, maxFlow(flow) ) );

    cv::imshow("horn_schunck_flow_rgb", flow_RGB);
    cv::imshow("flow_ground_truth_rgb", flowGroundTruth_RGB);

    std::cout << "Press any key to continue... \n" << std::endl;
    cv::waitKey(0);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Used to Read .flo files
// Arguments
// 1 - filePATH - the full path of the .flo file
// 2 - flowGroundTruth  - the matrix holding the outputed optical flow (optical flow) vectors for all pixels (dense optical flow), in the form of cv::Point2f
// 3 - maxxV    - contaims the maximum absolute flow value after the flow file is read

// (description of file structure - http://vision.middlebury.edu/flow/code/flow-code/README.txt)
// ".flo" file format used for optical flow evaluation
//
// Stores 2-band float image for horizontal (u) and vertical (v) flow components.
// Floats are stored in little-endian order.
// A flow value is considered "unknown" if either |u| or |v| is greater than 1e9.
//
//  bytes  contents
//
//  0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
//          (just a sanity check that floats are represented correctly)
//  4-7     width as an integer
//  8-11    height as an integer
//  12-end  data (width*height*2*4 bytes total)
//          the float values for u and v, interleaved, in row order, i.e.,
//          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...

void read_FLO_file( std::string filePATH, cv::Mat &flow, float &maxxV )
{
    std::cout << "read_FLO_file - " << filePATH << std::endl;
    std::ifstream myfile;
                  myfile.open (filePATH.data(), std::ios::binary);

    std::string inString(4,'\0');
    int         inInt;
    float       inFloat;

    float www = 0;
    float hhh = 0;

    std::cout << "fileSize      = " << sizeof(myfile) << std::endl;
    std::cout << "sizeof(int)   = " << sizeof(int  )  << std::endl;
    std::cout << "sizeof(float) = " << sizeof(float)  << std::endl;

    if(myfile.is_open())
    {
            myfile.read(        &inString[0], 4           );
            myfile.read( (char*)&inInt,       sizeof(int) );
            www = inInt;
            myfile.read( (char*)&inInt,       sizeof(int) );
            hhh = inInt;
    }

    //////////////////////////////////////////
    flow = cv::Mat::zeros(hhh,www,CV_32FC2);
    //////////////////////////////////////////

    maxxV = -999999;

    for (int iii=0; iii<hhh; iii++)
    {   for (int jjj=0; jjj<www; jjj++)
        {
            myfile.read( (char*)&inFloat, sizeof(float) );     flow.at<cv::Point2f>(iii,jjj).x = inFloat;
            myfile.read( (char*)&inFloat, sizeof(float) );     flow.at<cv::Point2f>(iii,jjj).y = inFloat;

            if ( flow.at<cv::Point2f>(iii,jjj).x<1e9 &&
                 flow.at<cv::Point2f>(iii,jjj).y<1e9 )
            {
                float magn = sqrt( pow(flow.at<cv::Point2f>(iii,jjj).x,2) +
                                   pow(flow.at<cv::Point2f>(iii,jjj).y,2) );

                if (magn>maxxV)  maxxV=magn;
            }
        }
    }

    std::cout << "\n\n~EOF~ " << maxxV << "\n\n" << std::endl;
    if(myfile.is_open())
    {
        myfile.read( (char*)&inFloat,     sizeof(float)  ); // just to reach EOF

        while(!myfile.eof())
        {
            std::cout << "not EOF!"  << std::endl;
        }
    }

    myfile.close();
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// borrowed & modified from
// Tobias Senst's post @
// http://stackoverflow.com/questions/20064818/how-to-draw-optical-flow-images-from-oclpyrlkopticalflowdense
// http://www.nue.tu-berlin.de/menue/mitarbeiter/tobias_senst

// Color-Code encoding for an optical flow matrix.
// Arguments
// 1 - inpFlow  - the matrix holding the outputed optical flow (optical flow) vectors for all pixels (dense optical flow), in the form of cv::Point2f
// 2 - max_size - output of the max optical flow vector magnitude (used elsewhere for visualization purposes)
// 3 - output   - (returned) - a CV_8UC3 cv::Mat containing the color-coded optical flow

cv::Mat flow_2_RGB( const cv::Mat &inpFlow, const float &max_size )
{

        //////////////////////////
        bool    use_value = false;
        cv::Mat sat;
        cv::Mat rgbFlow;
        //////////////////////////

        if (inpFlow.empty())                                                                                                             exit(1);
        if (inpFlow.depth() != CV_32F)  {   std::cout << "FlowTo RGB: error inpFlow wrong data type ( has be CV_32FC2" << std::endl;     exit(1);     }
        if (!sat.empty() )
            if( sat.type() != CV_8UC1)  {   std::cout << "FlowTo RGB: error sat must have type CV_8UC1"                << std::endl;     exit(1);     }

        const float grad2deg = (float)(90/3.141);
        double satMaxVal = 0;
        double minVal = 0;
        if(!sat.empty())
        {
            cv::minMaxLoc(sat, &minVal, &satMaxVal);
            satMaxVal = 255.0/satMaxVal;
        }

        cv::Mat pol(inpFlow.size(), CV_32FC2);

        float mean_val = 0, min_val = 1000, max_val = 0;

        for(int r = 0; r < inpFlow.rows; r++)
        {
            for(int c = 0; c < inpFlow.cols; c++)
            {
                cv::Mat1f inX( 1,1);  inX(0,0)=inpFlow.at<cv::Point2f>(r,c).x;
                cv::Mat1f inY( 1,1);  inY(0,0)=inpFlow.at<cv::Point2f>(r,c).y;
                cv::Mat1f outX(1,1);
                cv::Mat1f outY(1,1);
                cv::cartToPolar( inX, inY, outX, outY, false );
                cv::Point2f polar;
                            polar.x = outX(0,0);
                            polar.y = outY(0,0);

                polar.y *= grad2deg;
                mean_val +=polar.x;
                max_val = MAX(max_val, polar.x);
                min_val = MIN(min_val, polar.x);
                pol.at<cv::Point2f>(r,c) = cv::Point2f(polar.y,polar.x);
            }
        }
        mean_val /= inpFlow.size().area();
        float scale = max_val - min_val;
        float shift = -min_val;
        scale = 255.f/scale;
        if( max_size > 0)
        {
            scale = 255.f/max_size;
            shift = 0;
        }

        //calculate the angle, motion value
        cv::Mat hsv(inpFlow.size(), CV_8UC3);
        uchar * ptrHSV = hsv.ptr<uchar>();
        int idx_val  = (use_value) ? 2:1;
        int idx_sat  = (use_value) ? 1:2;


        for(int r = 0; r < inpFlow.rows; r++, ptrHSV += hsv.step1())
        {
            uchar * _ptrHSV = ptrHSV;
            for(int c = 0; c < inpFlow.cols; c++, _ptrHSV+=3)
            {
                cv::Point2f vpol = pol.at<cv::Point2f>(r,c);

                _ptrHSV[0] = cv::saturate_cast<uchar>(vpol.x);
                _ptrHSV[idx_val] = cv::saturate_cast<uchar>( (vpol.y + shift) * scale);
                if( sat.empty())
                    _ptrHSV[idx_sat] = 255;
                else
                    _ptrHSV[idx_sat] = 255-  cv::saturate_cast<uchar>(sat.at<uchar>(r,c) * satMaxVal);

            }
        }
        std::vector<cv::Mat> vec;
        cv::split(hsv, vec);
        cv::equalizeHist(vec[idx_val],vec[idx_val]);
        cv::merge(vec,hsv);
        cv::Mat rgbFlow32F;
        cv::cvtColor(hsv, rgbFlow32F, CV_HSV2BGR);
        rgbFlow32F.convertTo(rgbFlow, CV_8UC3);

        ///////////////
        return rgbFlow;
        ///////////////
}
