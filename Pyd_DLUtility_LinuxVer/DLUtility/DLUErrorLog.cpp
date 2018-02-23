//#include "stdafx.h"
#include "DLUErrorLog.h"

string cdlu::DLU_error_log(int errornum) {
    ostringstream OStr;
    OStr << "[Error - DLU" << std::setfill('0') << std::setw(3) << std::hex << errornum << std::setfill(' ')  << std::dec << "]: ";
    switch (errornum) {
    case 0x001:
        OStr << "[Proj1] The key/val of the projection list is not an np.ndarray.";
        break;
    case 0x002:
        OStr << "[Proj2] The size of the key/val of the projection list does not match the setting of projector.";
        break;
    case 0x003:
        OStr << "[Proj3] Unable to get the valid input data.";
        break;
    case 0x004:
        OStr << "[Proj4] The input channel does not match the setting of projector.";
        break;
    case 0x006:
        OStr << "[IOSes0] Should not call __read_log_info() before specifying the right path.";
        break;
    case 0x007:
        OStr << "[IOSes1] Called read position is out of range.";
        break;
    case 0x008:
        OStr << "[IOSes2] Indices should be a PySequence.";
        break;
    case 0x009:
        OStr << "[IOSes3] Could not extract the index from the indices sequence, please check your input.";
        break;
    case 0x00A:
        OStr << "[IOSes4] Need to input a shape like (height, weight).";
        break;
    case 0x100:
        OStr << "[Core00] The ordinary error raised by python core.";
        break;
    case 0x101:
        OStr << "[Core01] Fail to allocate an iterator of PyObject.";
        break;
    case 0x201:
        OStr << "[Npy01] Fail to allocate an iterator of PyArrayObject.";
        break;
    case 0x202:
        OStr << "[Npy02] Fail to allocate an iterating function of NpyIterator.";
        break;
    case 0x203:
        OStr << "[Npy03] Not a PyArrayObject which is intended to be gotten.";
        break;
    }
    return string(OStr.str());
}