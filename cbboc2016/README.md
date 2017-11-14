# How to cook

Download [CBBOC 2016 C++ API](//github.com/cbboc/cpp/) 

Add the following lines into "src/Main.cpp"
    
   CMAVNS2Competitor competitor;
   // CMAVNS2Competitor competitor( TrainingCategory::SHORT );
   // CMAVNS2Competitor competitor( TrainingCategory::LONG );
   CBBOC::run( competitor );
   
Install Eigen3 and libcmaes, then 

    g++ -o cmavns2016 src/Main.cpp -Iinclude -I/usr/include/eigen3 -std=c++0x -lstdc++ -lm -lcmaes -O3 -Wno-deprecated-declarations
