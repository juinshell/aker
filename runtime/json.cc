#include "json.h"
boost::property_tree::ptree ptr;


int get_kernel_info(const std::string &kernel_name, const std::string &key){
    return ptr.get_child(kernel_name).get<int>(key);
}

float fget_kernel_info(const std::string &kernel_name, const std::string &key){
    return ptr.get_child(kernel_name).get<float>(key);
}

void read_json(const std::string &filename)
{
  std::ifstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "Error: file not found" << std::endl;
    exit(1);
  }
  std::string str;
  std::string line;
  while (std::getline(file, line))
  {
    str += line;
  }
  file.close();
  ptr = boost::property_tree::ptree();
  std::stringstream ss;
  ss << str;
  boost::property_tree::read_json(ss, ptr);
}

