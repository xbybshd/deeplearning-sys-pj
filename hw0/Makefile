# NOTE: on MacOS you need to add an addition flag: -undefined dynamic_lookup
default:
	c++ -O3 -Wall -shared -std=c++11 -fPIC $$(python -m pybind11 --includes) src/simple_ml_ext.cpp -o src/simple_ml_ext.so
#c++ -O3 -Wall -shared -std=c++11 -fPIC -IC:\Users\pyg31\Anaconda3\Include -IC:\Users\pyg31\Anaconda3\lib\site-packages\pybind11\include src/simple_ml_ext.cpp -D_hypot=hypot -L"C:\\Users\\pyg31\\Anaconda3" -lpython37 -o src/simple_ml_ext.pyd
