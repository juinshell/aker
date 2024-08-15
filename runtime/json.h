#pragma once
#include <boost/json.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <stdarg.h>
using namespace boost::json;
const int JSON_NOT_FOUND = -INT16_MAX;
void read_json(const std::string &filename);

float fget_kernel_info(const std::string &kernel_name, const std::string &key);
int get_kernel_info(const std::string &kernel_name, const std::string &key);
void add_kernel_info(const std::string &kernel_name, const std::string &key, const int value);
void build_json(const std::string &filename);
std::string sget_kernel_info(const std::string &kernel_name, const std::string &key);

int geti(const std::string &key, ...);
float getf(const std::string &key, ...);
std::string gets(const std::string &key, ...);