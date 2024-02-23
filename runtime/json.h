#include <boost/json.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
using namespace boost::json;

void read_json(const std::string &filename);

float fget_kernel_info(const std::string &kernel_name, const std::string &key);
int get_kernel_info(const std::string &kernel_name, const std::string &key);