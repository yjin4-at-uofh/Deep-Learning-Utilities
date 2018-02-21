#ifndef DLUERRORLOG_H_INCLUDED
#define DLUERRORLOG_H_INCLUDED
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
using std::string;
using std::ostringstream;
using std::endl;
using std::cerr;
using std::cout;
namespace cdlu {
    extern string DLU_error_log(int errornum);
}
#endif