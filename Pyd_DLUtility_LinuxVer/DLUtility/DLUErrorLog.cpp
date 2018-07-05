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
        OStr << "[IOSes0] Should not call __read_log_info()/__write_log_info() before specifying the right path.";
        break;
    case 0x007:
        OStr << "[IO003] Called read position is out of range.";
        break;
    case 0x008:
        OStr << "[IO004] Indices should be a PySequence.";
        break;
    case 0x009:
        OStr << "[IO005] Could not extract the index from the indices sequence, please check your input.";
        break;
    case 0x00A:
        OStr << "[IOSes1] Need to input a shape like (height, weight).";
        break;
    case 0x00B:
        OStr << "[IOSes2] The input data should be arranged like a 2-dim array.";
        break;
    case 0x00C:
        OStr << "[IOSes3] Fail to convert a 2-dim array into C-data.";
        break;
    case 0x00D:
        OStr << "[IO001] Could not write data before calling save().";
        break;
    case 0x00E:
        OStr << "[IO002] Could not read data before calling load().";
        break;
    case 0x00F:
        OStr << "[IOFWM180602-1] The two databases (params and responses) do not match each other.";
        break;
    case 0x010:
        OStr << "[IO006] The number of the batch is invalid.";
        break;
    case 0x011:
        OStr << "[IO007] This IO type does not allow you to extract a slice from the whold data.";
        break;
    case 0x100:
        OStr << "[Core00] The ordinary error raised by python core.";
        break;
    case 0x101:
        OStr << "[Core01] Fail to allocate an iterator of PyObject.";
        break;
    case 0x102:
        OStr << "[Core02] The type of PyObject is not the required one.";
        break;
    case 0x103:
        OStr << "[Core03] Could not open a file by the read handle.";
        break;
    case 0x104:
        OStr << "[Core04] Could not open a file by the write handle.";
        break;
    case 0x105:
        OStr << "[Core05] Meet fatal errors while I/O a file.";
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
    case 0x204:
        OStr << "[Npy04] Could not extract the index from the sequence, please check your input.";
        break;
    case 0x205:
        OStr << "[Npy05] The input data should be arranged like a 2-dim array.";
        break;
    case 0x206:
        OStr << "[Npy06] Fail to convert an ndarray into C-data.";
        break;
    case 0x207:
        OStr << "[Npy07] The input data should be arranged like a 1-dim array (vector).";
        break;
    case 0x208:
        OStr << "[Npy08] The data type of the array does not match the intended one.";
        break;
    case 0x209:
        OStr << "[Npy09] Fail to build an ndarray from C-data.";
        break;
    }
    return string(OStr.str());
}