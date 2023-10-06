#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;


class OtherClass {       // The class
  public:             // Access specifier
    int rnd;        // Attribute (int variable)
    string str;  // Attribute (string variable)
    void update();    
};


class MyClass {       // The class
  public:             // Access specifier
    int myNum;        // Attribute (int variable)
    string myString;  // Attribute (string variable)
    void update();
    OtherClass* subclass;
    thrust::host_vector<int> my_vector;

};

int main() {
  MyClass myObj;  // Create an object of MyClass

  // Access attributes and set values
  myObj.myNum = 1; 
  myObj.subclass = new(OtherClass);
  myObj.myString = "Some text";
  myObj.my_vector.resize(0);
  myObj.my_vector.push_back(39879);

  myObj.update();

  // Print attribute values
  cout << myObj.myNum << myObj.myString << std::endl;
  cout << myObj.subclass->rnd << myObj.subclass->str << std::endl;
  cout << myObj.my_vector[0]<< std::endl;
  return 0;
}

void MyClass::update(){
    myNum  = 69;
    myString = "More text";

    subclass->rnd  = 420;
    subclass->str = "subtext";

    my_vector[0] = 222;
}