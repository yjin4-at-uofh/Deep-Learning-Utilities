//#include "stdafx.h"
#include "DLUSettings.h"
cdlu::IO_Abstract::IO_Abstract(void) {
    __h.clear();
    __oh.clear();
}
cdlu::IO_Abstract::~IO_Abstract(void) {
    clear();
}
cdlu::IO_Abstract::IO_Abstract(IO_Abstract &&ref) noexcept {
    __h = std::move(ref.__h);
    __oh = std::move(ref.__oh);
}
cdlu::IO_Abstract& cdlu::IO_Abstract::operator=(IO_Abstract &&ref) noexcept {
    if (this != &ref) {
        __oh = std::move(ref.__oh);
        __h = std::move(ref.__h);
    }
    return *this;
}
std::ostream & cdlu::operator<<(std::ostream & out, const cdlu::IO_Abstract & self_class) {
    return self_class.__print(out);
}
void cdlu::IO_Abstract::clear() {
    if (__h.is_open()) {
        __h.clear();
        __h.close();
    }
    if (__oh.is_open()) {
        __oh.clear();
        __oh.close();
    }
}
PyObject *cdlu::IO_Abstract::size() const {
    return nullptr;
}
PyObject * cdlu::IO_Abstract::read(size_t s_num) {
    return nullptr;
}
PyObject * cdlu::IO_Abstract::read(PyObject *s_numPyList) {
    return nullptr;
}
PyObject * cdlu::IO_Abstract::read(int batchNum, PyObject *batchShape) {
    return nullptr;
}
int64_t cdlu::IO_Abstract::write(PyObject *data) {
    return -1;
}
std::ostream & cdlu::IO_Abstract::__print(std::ostream & out) const {
    auto self_size = size();
    out << "<IOHandle - Abstract:" << endl;
    if (__h.is_open()) {
        out << "     IN_Handle: opened" << endl;
    }
    else {
        out << "     IN_Handle: closed" << endl;
    }
    if (__oh.is_open()) {
        out << "    OUT_Handle: opened" << endl;
    }
    else {
        out << "    OUT_Handle: closed" << endl;
    }
    out << ">";
    return out;
}

cdlu::IO_Sesmic::IO_Sesmic(void):
    num_shot(0), num_rec(0), num_time(0), onum_shot(0), onum_rec(0), onum_time(0), 
    IO_Abstract() {
    __filename.clear();
    __folderpath.clear();
    __ofilename.clear();
    __ofolderpath.clear();
}
cdlu::IO_Sesmic::~IO_Sesmic(void) {
}
cdlu::IO_Sesmic::IO_Sesmic(const IO_Sesmic &ref):
    IO_Sesmic() {
    auto refname = ref.__filename;
    if (!refname.empty() && ref.__h.is_open()) {
        if (!ref.__folderpath.empty()) {
            refname.insert(0, "/");
            refname.insert(0, ref.__folderpath);
        }
        load(refname);
    }
    refname = ref.__ofilename;
    if (!refname.empty() && ref.__oh.is_open()) {
        if (!ref.__ofolderpath.empty()) {
            refname.insert(0, "/");
            refname.insert(0, ref.__ofolderpath);
        }
        save(refname);
    }
}
cdlu::IO_Sesmic& cdlu::IO_Sesmic::operator=(const IO_Sesmic &ref) {
    if (this != &ref) {
        auto refname = ref.__filename;
        if (!refname.empty() && ref.__h.is_open()) {
            if (!ref.__folderpath.empty()) {
                refname.insert(0, "/");
                refname.insert(0, ref.__folderpath);
            }
            load(refname);
        }
        refname = ref.__ofilename;
        if (!refname.empty() && ref.__oh.is_open()) {
            if (!ref.__ofolderpath.empty()) {
                refname.insert(0, "/");
                refname.insert(0, ref.__ofolderpath);
            }
            save(refname);
        }
    }
    return *this;
}
cdlu::IO_Sesmic::IO_Sesmic(IO_Sesmic &&ref) noexcept:
    num_shot(ref.num_shot), num_rec(ref.num_rec), num_time(ref.num_time),
    onum_shot(ref.onum_shot), onum_rec(ref.onum_rec), onum_time(ref.onum_time),
    IO_Abstract(std::move(ref)) {
    __filename = std::move(ref.__filename);
    __folderpath = std::move(ref.__folderpath);
    __ofilename = std::move(ref.__ofilename);
    __ofolderpath = std::move(ref.__ofolderpath);
}
cdlu::IO_Sesmic& cdlu::IO_Sesmic::operator=(IO_Sesmic &&ref) noexcept {
    if (this != &ref) {
        num_shot = ref.num_shot;
        num_rec = ref.num_rec;
        num_time = ref.num_time;
        onum_shot = ref.onum_shot;
        onum_rec = ref.onum_rec;
        onum_time = ref.onum_time;
        __filename = std::move(ref.__filename);
        __folderpath = std::move(ref.__folderpath);
        __ofilename = std::move(ref.__ofilename);
        __ofolderpath = std::move(ref.__ofolderpath);
        __oh = std::move(ref.__oh);
        __h = std::move(ref.__h);
    }
    return *this;
}
std::ostream & cdlu::operator<<(std::ostream & out, const cdlu::IO_Sesmic & self_class) {
    return self_class.__print(out);
}
void cdlu::IO_Sesmic::clear() {
    bool protectWrite = __oh.is_open();
    IO_Abstract::clear();
    if (protectWrite) {
        __write_log_info();
    }
    __filename.clear();
    __folderpath.clear();
    __ofilename.clear();
    __ofolderpath.clear();
    num_shot = 0;
    num_rec = 0;
    num_time = 0;
    onum_shot = 0;
    onum_rec = 0;
    onum_time = 0;
}
PyObject *cdlu::IO_Sesmic::size() const {
    PyObject *pysize = Py_BuildValue("[iii]", num_shot, num_rec, num_time);
    return pysize;
}
bool cdlu::IO_Sesmic::save(string filename) {
    auto pathpos_A = filename.rfind('/');
    auto pathpos_B = filename.rfind('\\');
    if (pathpos_A != string::npos && pathpos_B != string::npos) {
        pathpos_A = max(pathpos_A, pathpos_B);
    }
    else if (pathpos_B != string::npos) {
        pathpos_A = pathpos_B;
    }

    if (pathpos_A != string::npos) {
        __ofolderpath = filename.substr(0, pathpos_A);
        __ofilename = filename.substr(pathpos_A + 1);
    }
    else {
        __ofolderpath.clear();
        __ofilename = filename;
    }
    string binname(std::move(__full_path(true)));
    __oh.open(binname, std::ios::out | std::ofstream::binary);
    if (__oh.fail()) {
        cerr << cdlu::DLU_error_log(0x006) << ", fail to open: \"" << binname << "\"" << endl;
        __oh.clear();
        clear();
        return false;
    }
    onum_rec = 0;
    onum_shot = 0;
    onum_time = 0;
    return true;
}
bool cdlu::IO_Sesmic::load(string filename) {
    auto pathpos_A = filename.rfind('/');
    auto pathpos_B = filename.rfind('\\');
    if (pathpos_A != string::npos && pathpos_B != string::npos) {
        pathpos_A = max(pathpos_A, pathpos_B);
    }
    else if (pathpos_B != string::npos) {
        pathpos_A = pathpos_B;
    }
    
    if (pathpos_A != string::npos) {
        __folderpath = filename.substr(0, pathpos_A);
        __filename = filename.substr(pathpos_A + 1);
    }
    else {
        __folderpath.clear();
        __filename = filename;
    }
    if (!__read_log_info()) {
        clear();
        return false;
    }
    string binname(std::move(__full_path()));
    __h.open(binname, std::ifstream::binary);
    if (__h.fail()) {
        cerr << cdlu::DLU_error_log(0x006) << ", fail to open: \"" << binname << "\"" << endl;
        __h.clear();
        clear();
        return false;
    }
    return true;
}
void cdlu::IO_Sesmic::close() {
    clear();
}
PyObject *cdlu::IO_Sesmic::read(size_t s_num) {
    if (!__h.is_open()) {
        cerr << cdlu::DLU_error_log(0x00E) << endl;
        return nullptr;
    }
    if (PyArray_API == nullptr) {
        import_array();
    }
    if (s_num >= num_shot) {
        cerr << cdlu::DLU_error_log(0x007) << ", max position=" << num_shot-1 << endl;
        return nullptr;
    }
    auto thisSize = num_rec * num_time;
    auto offset = s_num * thisSize * sizeof(float);
    __h.clear();
    __h.seekg(offset, std::ios::beg);
    auto out_data = new float[thisSize];
    __h.read(reinterpret_cast<char *>(out_data), thisSize * sizeof(float));
    npy_intp odims[] = {static_cast<npy_intp>(num_rec), static_cast<npy_intp>(num_time) };
    PyObject *PyResPic = PyArray_SimpleNewFromData(2, odims, NPY_FLOAT32, reinterpret_cast<void *>(out_data));
    PyArray_ENABLEFLAGS((PyArrayObject *)PyResPic, NPY_ARRAY_OWNDATA);
    return PyResPic;
}
PyObject *cdlu::IO_Sesmic::read(PyObject *s_numPyList) {
    if (!__h.is_open()) {
        cerr << cdlu::DLU_error_log(0x00E) << endl;
        return nullptr;
    }
    if (PyArray_API == nullptr) {
        import_array();
    }
    auto ochannels = PySequence_Size(s_numPyList);
    auto s_numPyList_F = PySequence_Fast(s_numPyList, "Fail to get access to the index list.");
    try {
        if (ochannels == -1) {
            throw 0x008;
        }
        auto longcheck = 0;
        for (decltype(ochannels) i = 0; i < ochannels; i++) {
            auto f_obj = PySequence_Fast_GET_ITEM(s_numPyList, i);
            if (!f_obj) {
                throw 0x009;
            }
            auto f_pos = PyLong_AsLongAndOverflow(f_obj, &longcheck);
            if (longcheck) {
                throw 0x009;
            }
            else if (f_pos >= num_shot) {
                cerr << "Error occurrs at channel " << i << ", max position=" << num_shot - 1 << endl;
                throw 0x007;
            }
        }
    }
    catch (int errornum) {
        cerr << cdlu::DLU_error_log(errornum) << endl;
        Py_XDECREF(s_numPyList_F);
        return nullptr;
    }
    auto sliceSize = num_rec * num_time;
    auto dataSize = sliceSize * ochannels;
    auto out_data = new float[dataSize];
    for (decltype(ochannels) i = 0; i < ochannels; i++) {
        auto f_pos = PyLong_AsLong(PySequence_Fast_GET_ITEM(s_numPyList, i)) * sliceSize * sizeof(float);
        __h.clear();
        __h.seekg(f_pos, std::ios::beg);
        for (auto *p = out_data+i; p < out_data + dataSize; p += ochannels) {
            __h.read(reinterpret_cast<char *>(p), sizeof(float));
        }
    }
    npy_intp odims[] = { static_cast<npy_intp>(num_rec), static_cast<npy_intp>(num_time), static_cast<npy_intp>(ochannels) };
    PyObject *PyResPic = PyArray_SimpleNewFromData(3, odims, NPY_FLOAT32, reinterpret_cast<void *>(out_data));
    PyArray_ENABLEFLAGS((PyArrayObject *)PyResPic, NPY_ARRAY_OWNDATA);
    Py_XDECREF(s_numPyList_F);
    return PyResPic;
}
PyObject *cdlu::IO_Sesmic::read(int batchNum, PyObject *batchShape) {
    if (batchNum <= 0) {
        Py_RETURN_NONE;
    }
    if (!__h.is_open()) {
        cerr << cdlu::DLU_error_log(0x00E) << endl;
        Py_RETURN_NONE;
    }
    if (PyArray_API == nullptr) {
        import_array();
    }
    auto ochannels = PySequence_Size(batchShape);
    auto batchShape_F = PySequence_Fast(batchShape, "Fail to get access to the (H, W).");
    try {
        if (ochannels != 2) {
            throw 0x00A;
        }
        auto longcheck = 0;
        for (decltype(ochannels) i = 0; i < ochannels; i++) {
            auto f_obj = PySequence_Fast_GET_ITEM(batchShape_F, i);
            if (!f_obj) {
                throw 0x009; 
            }
            auto f_pos = PyLong_AsLongAndOverflow(f_obj, &longcheck);
            if (longcheck) {
                throw 0x009;
            }
            else if (i == 0 && f_pos >= num_rec) {
                cerr << "Error occurrs at height= " << static_cast<int>(f_pos) << ", max height=" << num_rec << endl;
                throw 0x007;
            }
            else if (i == 1 && f_pos >= num_time) {
                cerr << "Error occurrs at width= " << static_cast<int>(f_pos) << ", max width=" << num_time << endl;
                throw 0x007;
            }
        }
    }
    catch (int errornum) {
        cerr << cdlu::DLU_error_log(errornum) << endl;
        Py_XDECREF(batchShape_F);
        Py_RETURN_NONE;
    }
    auto h = PyLong_AsLong(PySequence_Fast_GET_ITEM(batchShape_F, 0));
    auto w = PyLong_AsLong(PySequence_Fast_GET_ITEM(batchShape_F, 1));
    auto offset_fig = num_time * num_rec;
    auto offset_row = num_time;
    std::default_random_engine rd_e(rand());
    std::uniform_int_distribution<size_t> rd_shot(0, num_shot - 1);
    decltype(rd_shot) rd_w(0, num_time - w);
    decltype(rd_shot) rd_h(0, num_rec - h);
    auto newdata = new float[h*w*batchNum];
    auto p = newdata;
    size_t chunkStart = 0;
    for (int i = 0; i < batchNum; i++) {
        chunkStart = rd_shot(rd_e)*offset_fig + rd_h(rd_e)*offset_row + rd_w(rd_e);
        for (size_t row = 0; row < h; row++, p += w) {
            __h.clear();
            __h.seekg((chunkStart + row * offset_row) * sizeof(float), std::ios::beg); // The matrix is stored in a row order.
            __h.read(reinterpret_cast<char *>(p), w * sizeof(float));
        }
    }
    npy_intp odims[] = { static_cast<npy_intp>(batchNum), static_cast<npy_intp>(h), static_cast<npy_intp>(w), 1 };
    PyObject *PyResPic = PyArray_SimpleNewFromData(4, odims, NPY_FLOAT32, reinterpret_cast<void *>(newdata));
    PyArray_ENABLEFLAGS((PyArrayObject *)PyResPic, NPY_ARRAY_OWNDATA);
    Py_XDECREF(batchShape_F);
    return PyResPic;
}
int64_t cdlu::IO_Sesmic::write(PyObject *data) {
    if (PyArray_API == nullptr) {
        import_array();
    }
    int dim_n = 0;
    npy_intp *dims = nullptr;
    try {
        if (!__oh.is_open()) {
            throw 0x00D;
        }
        dim_n = PyArray_NDIM((PyArrayObject *)data);
        if (dim_n != 2) {
            throw 0x00B;
        }
        dims = PyArray_SHAPE((PyArrayObject *)data);
        if (onum_rec > 0) {
            if (onum_rec != dims[0]) {
                cerr << "Current receiver number " << static_cast<int>(dims[0]) << " does not correspond to the series: " << static_cast<int>(onum_rec) << endl;
                throw 0x00C;
            }
        }
        else {
            onum_rec = dims[0];
        }
        if (onum_time > 0) {
            if (onum_time != dims[1]) {
                cerr << "Current receiver time step " << static_cast<int>(dims[1]) << " does not correspond to the series: " << static_cast<int>(onum_time) << endl;
                throw 0x00B;
            }
        }
        else {
            onum_time = dims[1];
        }
    }
    catch (int errornum) {
        cerr << cdlu::DLU_error_log(errornum) << endl;
        return -1;
    }
    npy_intp numels[1] = {dims[0]*dims[1]};
    auto descr = PyArray_DescrFromType(NPY_FLOAT32);
    auto n_data = PyArray_FromArray((PyArrayObject*)data, descr, NPY_ARRAY_C_CONTIGUOUS);
    if (n_data == nullptr) {
        cerr << cdlu::DLU_error_log(0x00C) << endl;
        return -1;
    }
    auto writebytes = static_cast<size_t>(numels[0]) * sizeof(float);
    __oh.write(reinterpret_cast<const char *>(PyArray_DATA((PyArrayObject*)n_data)), writebytes);
    if (__oh.good()) {
        onum_shot++;
    }
    else {
        return -1;
    }
    PyArray_XDECREF((PyArrayObject*)n_data);
    return writebytes;
}
bool cdlu::IO_Sesmic::__write_log_info() {
    if (__ofilename.empty()) {
        cerr << cdlu::DLU_error_log(0x006) << endl;
        return false;
    }
    string logname(__ofilename);
    logname.insert(0, "TDATA_");
    logname.append(".LOG");
    if (!__ofolderpath.empty()) {
        logname.insert(0, "/");
        logname.insert(0, __ofolderpath);
    }
    __oh.open(logname, std::ios::out);
    if (__oh.fail()) {
        cerr << cdlu::DLU_error_log(0x006) << ", fail to open: \"" << logname << "\"" << endl;
        __oh.clear();
        return false;
    }
    __oh << "Number of time steps: " << onum_time << endl;
    __oh << "Number of shots: " << onum_shot << endl;
    __oh << "Maximum number of receivers: " << onum_rec << endl;
    __oh.clear();
    __oh.close();
    return true;
}
bool cdlu::IO_Sesmic::__read_log_info() {
    if (__filename.empty()){
        cerr << cdlu::DLU_error_log(0x006) << endl;
        return false;
    }
    string logname(__filename);
    logname.insert(0, "TDATA_");
    logname.append(".LOG");
    if (!__folderpath.empty()) {
        logname.insert(0, "/");
        logname.insert(0, __folderpath);
    }
    __h.open(logname, std::ios::in);
    if (__h.fail()) {
        cerr << cdlu::DLU_error_log(0x006) << ", fail to open: \"" << logname  << "\"" << endl;
        __h.clear();
        return false;
    }
    string s, key;
    std::istringstream in_s;

    while (std::getline(__h, s)) {
        key.assign("Number of time steps:");
        auto pos = s.find(key);
        if (pos != string::npos) {
            auto sub_str = s.substr(pos + key.length());
            in_s.clear();
            in_s.str(sub_str);
            in_s >> num_time;
        }
        else {
            key.assign("Number of shots:");
            pos = s.find(key);
            if (pos != string::npos) {
                auto sub_str = s.substr(pos + key.length());
                in_s.clear();
                in_s.str(sub_str);
                in_s >> num_shot;
            }
            else {
                key.assign("Maximum number of receivers:");
                pos = s.find(key);
                if (pos != string::npos) {
                    auto sub_str = s.substr(pos + key.length());
                    in_s.clear();
                    in_s.str(sub_str);
                    in_s >> num_rec;
                }
            }
        }
    }
    __h.clear();
    __h.close();
    return true;
}
string cdlu::IO_Sesmic::__full_path(bool write) {
    string binname;
    if (write) {
        binname.assign(__ofilename);
    }
    else{
        binname.assign(__filename);
    }
    binname.insert(0, "TDATA_");
    binname.append(".BIN");
    if (write) {
        if (!__ofolderpath.empty()) {
            binname.insert(0, "/");
            binname.insert(0, __ofolderpath);
        }
    }
    else {
        if (!__folderpath.empty()) {
            binname.insert(0, "/");
            binname.insert(0, __folderpath);
        }
    }
    return binname;
}
std::ostream & cdlu::IO_Sesmic::__print(std::ostream & out) const {
    auto self_size = size();
    out << "<IOHandle - Sesmic:" << endl;
    auto empty = true;
    if (__h.is_open()) {
        empty = false;
        out << "    |-i:          folder=" << __folderpath << endl;
        out << "    |-i:            name=" << __filename << endl;
        out << "    |-i:   num. of shots=" << num_shot << endl;
        out << "    |-i:   num. of recv.=" << num_rec << endl;
        out << "    |-i:  num. of tstep.=" << num_time << endl;
    }
    if (__oh.is_open()) {
        empty = false;
        out << "    |-o:          folder=" << __ofolderpath << endl;
        out << "    |-o:            name=" << __ofilename << endl;
        out << "    |-o:   num. of shots=" << onum_shot << endl;
        out << "    |-o:   num. of recv.=" << onum_rec << endl;
        out << "    |-o:  num. of tstep.=" << onum_time << endl;
    }
    if (empty) {
        out << "    empty" << endl;
    }
    out << ">";
    return out;
}

cdlu::IO_FWM180602::IO_FWM180602(void) :
    evalSize(0), num_m(0), i_base(0), o_base(0), IO_Abstract() {
    __filename.clear();
    __h_d.clear();
}
cdlu::IO_FWM180602::~IO_FWM180602(void) {
    if (__h_d.is_open()) {
        __h_d.clear();
        __h_d.close();
    }
}
cdlu::IO_FWM180602::IO_FWM180602(const IO_FWM180602 &ref) :
    IO_FWM180602() {
    auto refname = ref.__filename;
    if (!refname.empty()) {
        if (!refname.empty() && ref.__h.is_open()) {
            load(refname);
        } 
    }
}
cdlu::IO_FWM180602& cdlu::IO_FWM180602::operator=(const IO_FWM180602 &ref) {
    if (this != &ref) {
        auto refname = ref.__filename;
        if (!refname.empty()) {
            if (!refname.empty() && ref.__h.is_open()) {
                load(refname);
            }
        }
    }
    return *this;
}
cdlu::IO_FWM180602::IO_FWM180602(IO_FWM180602 &&ref) noexcept:
evalSize(ref.evalSize), num_m(ref.num_m), i_base(ref.i_base), o_base(ref.o_base), IO_Abstract(std::move(ref)) {
    __filename = std::move(ref.__filename);
    __h_d = std::move(ref.__h_d);
}
cdlu::IO_FWM180602& cdlu::IO_FWM180602::operator=(IO_FWM180602 &&ref) noexcept {
    if (this != &ref) {
        evalSize = ref.evalSize;
        num_m = ref.num_m;
        i_base = ref.i_base;
        o_base = ref.o_base;
        __filename = std::move(ref.__filename);
        __oh = std::move(ref.__oh);
        __h = std::move(ref.__h);
        __h_d = std::move(ref.__h_d);
    }
    return *this;
}
std::ostream & cdlu::operator<<(std::ostream & out, const cdlu::IO_FWM180602 & self_class) {
    return self_class.__print(out);
}
void cdlu::IO_FWM180602::clear() {
    IO_Abstract::clear();
    __filename.clear();
    evalSize = 0;
    num_m = 0;
    i_base = 0;
    o_base = 0;
    if (__h_d.is_open()) {
        __h_d.clear();
        __h_d.close();
    }
}
PyObject *cdlu::IO_FWM180602::size() const {
    PyObject *pysize = Py_BuildValue("[ii]", evalSize, num_m);
    return pysize;
}
bool cdlu::IO_FWM180602::load(string filename) {
    __filename = filename;
    auto para_filename = filename + ".fwdp";
    auto resp_filename = filename + ".fwdr";
    try {
        decltype(evalSize) checkEvalSize = 0;
        decltype(num_m) checkNumM = 0;
        __h.open(para_filename, std::ifstream::in | std::ifstream::binary);
        if (!__h.good()) {
            cerr << "Fail to open: \"" << para_filename << "\"" << endl;
            throw 0x103;
        }
        __h_d.open(resp_filename, std::ifstream::in | std::ifstream::binary);
        if (!__h.good()) {
            cerr << "Fail to open: \"" << resp_filename << "\"" << endl;
            throw 0x103;
        }
        __h.read(reinterpret_cast<char *>(&evalSize), sizeof(decltype(evalSize)));
        __h.read(reinterpret_cast<char *>(&num_m), sizeof(decltype(num_m)));
        __h_d.read(reinterpret_cast<char *>(&checkEvalSize), sizeof(decltype(checkEvalSize)));
        __h_d.read(reinterpret_cast<char *>(&checkNumM), sizeof(decltype(checkNumM)));
        if (!__h.good() || !__h_d.good())
            throw 0x105;
        if (evalSize != checkEvalSize || num_m != checkNumM)
            throw 0x00F;
        i_base = __h.tellg() + static_cast<decltype(i_base)>(num_m * (2 * sizeof(DataType) + sizeof(size_t)));
        o_base = __h_d.tellg();
    }
    catch (int errornum) {
        cerr << cdlu::DLU_error_log(errornum) << endl;
        clear();
        return false;
    }
    return true;
}
void cdlu::IO_FWM180602::close() {
    clear();
}
PyObject *cdlu::IO_FWM180602::read(size_t s_num) {
    if (!__h.is_open() || !__h_d.is_open()) {
        cerr << cdlu::DLU_error_log(0x00E) << endl;
        return nullptr;
    }
    if (PyArray_API == nullptr) {
        import_array();
    }
    PyObject *PyRes_i = nullptr;
    PyObject *PyRes_o = nullptr;
    DataType *out_data_i = nullptr;
    DataType *out_data_o = nullptr;
    try {
        if (s_num >= evalSize) {
            cerr << "Max position=" << evalSize - 1 << endl;
            throw 0x007;
        }
        auto offset_i = i_base + static_cast<decltype(i_base)>(s_num * num_m * sizeof(DataType));
        auto offset_o = o_base + static_cast<decltype(o_base)>(s_num * FWDCURVES180602_NUMPOINTS * sizeof(DataType));
        // Build the data from the geophysical database.
        __h.clear();
        __h.seekg(offset_i, std::ios::beg);
        out_data_i = new DataType[num_m];
        __h.read(reinterpret_cast<char *>(out_data_i), num_m * sizeof(DataType));
        if (!__h.good())
            throw 0x105;
        npy_intp odims_i[] = { 1, static_cast<npy_intp>(num_m) };
        PyRes_i = PyArray_SimpleNewFromData(2, odims_i, FWDCURVES180602_DATATYPE, reinterpret_cast<void *>(out_data_i));
        if (PyRes_i == nullptr) {
            throw 0x209;
        }
        PyArray_ENABLEFLAGS((PyArrayObject *)PyRes_i, NPY_ARRAY_OWNDATA);
        // Build the data from the response database.
        __h_d.clear();
        __h_d.seekg(offset_o, std::ios::beg);
        out_data_o = new DataType[FWDCURVES180602_NUMPOINTS];
        __h_d.read(reinterpret_cast<char *>(out_data_o), FWDCURVES180602_NUMPOINTS * sizeof(DataType));
        if (!__h_d.good())
            throw 0x105;
        npy_intp odims_o[] = { 1, FWDCURVES180602_NUMPOINTS };
        PyRes_o = PyArray_SimpleNewFromData(2, odims_o, FWDCURVES180602_DATATYPE, reinterpret_cast<void *>(out_data_o));
        if (PyRes_o == nullptr) {
            throw 0x209;
        }
        PyArray_ENABLEFLAGS((PyArrayObject *)PyRes_o, NPY_ARRAY_OWNDATA);
    }
    catch (int errornum) {
        cerr << cdlu::DLU_error_log(errornum) << endl;
        if (PyRes_i){
            Py_XDECREF(PyRes_i);
        }
        else if (out_data_i){
            delete[] out_data_i;
        }
        if (PyRes_o) {
            Py_XDECREF(PyRes_o);
        }
        else if (out_data_o) {
            delete[] out_data_o;
        }
        __h.clear();
        __h_d.clear();
        return nullptr;
    }
    return Py_BuildValue("(OO)", PyRes_i, PyRes_o);
}
PyObject *cdlu::IO_FWM180602::read(PyObject *s_numPyList) {
    if (!__h.is_open() || !__h_d.is_open()) {
        cerr << cdlu::DLU_error_log(0x00E) << endl;
        return nullptr;
    }
    if (PyArray_API == nullptr) {
        import_array();
    }
    Py_ssize_t ochannels;
    auto s_numPyList_F = PySequence_Fast(s_numPyList, "Fail to get access to the index list.");
    try {
        ochannels = PySequence_Size(s_numPyList);
        if (ochannels == -1) {
            throw 0x008;
        }
        auto longcheck = 0;
        for (decltype(ochannels) i = 0; i < ochannels; i++) {
            auto f_obj = PySequence_Fast_GET_ITEM(s_numPyList_F, i);
            if (!f_obj) {
                throw 0x009;
            }
            auto f_pos = PyLong_AsLongAndOverflow(f_obj, &longcheck);
            if (longcheck) {
                throw 0x009;
            }
            else if (f_pos >= evalSize) {
                cerr << "Error occurrs at channel " << i << ", max position=" << evalSize - 1 << endl;
                throw 0x007;
            }
        }
    }
    catch (int errornum) {
        cerr << cdlu::DLU_error_log(errornum) << endl;
        Py_XDECREF(s_numPyList_F);
        return nullptr;
    }
    PyObject *PyRes_i = nullptr;
    PyObject *PyRes_o = nullptr;
    DataType *out_data_i = nullptr;
    DataType *out_data_o = nullptr;
    try {
        // Build the data from the geophysical database.
        out_data_i = new DataType[num_m*ochannels];
        for (decltype(ochannels) i = 0; i < ochannels; i++) {
            auto cur_data = out_data_i + i * num_m;
            auto f_pos = i_base + static_cast<decltype(i_base)>(PyLong_AsLong(PySequence_Fast_GET_ITEM(s_numPyList_F, i)) * num_m * sizeof(DataType));
            __h.clear();
            __h.seekg(f_pos, std::ios::beg);
            __h.read(reinterpret_cast<char *>(cur_data), num_m * sizeof(DataType));
            if (!__h.good())
                throw 0x105;
        }
        npy_intp odims_i[] = { static_cast<npy_intp>(ochannels), static_cast<npy_intp>(num_m) };
        PyRes_i = PyArray_SimpleNewFromData(2, odims_i, FWDCURVES180602_DATATYPE, reinterpret_cast<void *>(out_data_i));
        if (PyRes_i == nullptr) {
            throw 0x209;
        }
        PyArray_ENABLEFLAGS((PyArrayObject *)PyRes_i, NPY_ARRAY_OWNDATA);
        // Build the data from the response database.
        out_data_o = new DataType[FWDCURVES180602_NUMPOINTS*ochannels];
        for (decltype(ochannels) i = 0; i < ochannels; i++) {
            auto cur_data = out_data_o + i * FWDCURVES180602_NUMPOINTS;
            auto f_pos = o_base + static_cast<decltype(o_base)>(PyLong_AsLong(PySequence_Fast_GET_ITEM(s_numPyList_F, i)) * FWDCURVES180602_NUMPOINTS * sizeof(DataType));
            __h_d.clear();
            __h_d.seekg(f_pos, std::ios::beg);
            __h_d.read(reinterpret_cast<char *>(cur_data), FWDCURVES180602_NUMPOINTS * sizeof(DataType));
            if (!__h_d.good())
                throw 0x105;
        }
        npy_intp odims_o[] = { static_cast<npy_intp>(ochannels), FWDCURVES180602_NUMPOINTS };
        PyRes_o = PyArray_SimpleNewFromData(2, odims_o, FWDCURVES180602_DATATYPE, reinterpret_cast<void *>(out_data_o));
        if (PyRes_o == nullptr) {
            throw 0x209;
        }
        PyArray_ENABLEFLAGS((PyArrayObject *)PyRes_o, NPY_ARRAY_OWNDATA);
    }
    catch (int errornum) {
        cerr << cdlu::DLU_error_log(errornum) << endl;
        if (PyRes_i) {
            Py_XDECREF(PyRes_i);
        }
        else if (out_data_i) {
            delete[] out_data_i;
        }
        if (PyRes_o) {
            Py_XDECREF(PyRes_o);
        }
        else if (out_data_o) {
            delete[] out_data_o;
        }
        Py_XDECREF(s_numPyList_F);
        __h.clear();
        __h_d.clear();
        return nullptr;
    }
    Py_XDECREF(s_numPyList_F);
    return Py_BuildValue("(OO)", PyRes_i, PyRes_o);
}
PyObject *cdlu::IO_FWM180602::read(int batchNum, PyObject *batchShape) { // Note batchShape would not be used here.
    try {
        if (batchNum <= 0)
            throw 0x010;
        if (batchShape != Py_None) 
            throw 0x011;
        if (!__h.is_open() || !__h_d.is_open())
            throw 0x00E;
    }
    catch (int errornum) {
        cerr << cdlu::DLU_error_log(errornum) << endl;
        return nullptr;
    }
    if (PyArray_API == nullptr) {
        import_array();
    }
    PyObject *PyRes_i = nullptr;
    PyObject *PyRes_o = nullptr;
    DataType *out_data_i = nullptr;
    DataType *out_data_o = nullptr;
    try {
        std::default_random_engine rd_e(rand());
        std::uniform_int_distribution<size_t> rd_ind(0, evalSize - 1);
        std::shared_ptr<size_t []> indList(new size_t[batchNum]);
        auto indList_p = indList.get();
        for (auto i = 0; i < batchNum; i++) {
            indList_p[i] = rd_ind(rd_e);
        }
        // Build the data from the geophysical database.
        out_data_i = new DataType[num_m*batchNum];
        for (decltype(batchNum) i = 0; i < batchNum; i++) {
            auto cur_data = out_data_i + i * num_m;
            auto f_pos = i_base + static_cast<decltype(i_base)>(indList_p[i] * num_m * sizeof(DataType));
            __h.clear();
            __h.seekg(f_pos, std::ios::beg);
            __h.read(reinterpret_cast<char *>(cur_data), num_m * sizeof(DataType));
            if (!__h.good())
                throw 0x105;
        }
        npy_intp odims_i[] = { static_cast<npy_intp>(batchNum), static_cast<npy_intp>(num_m) };
        PyRes_i = PyArray_SimpleNewFromData(2, odims_i, FWDCURVES180602_DATATYPE, reinterpret_cast<void *>(out_data_i));
        if (PyRes_i == nullptr) {
            throw 0x209;
        }
        PyArray_ENABLEFLAGS((PyArrayObject *)PyRes_i, NPY_ARRAY_OWNDATA);
        // Build the data from the response database.
        out_data_o = new DataType[FWDCURVES180602_NUMPOINTS*batchNum];
        for (decltype(batchNum) i = 0; i < batchNum; i++) {
            auto cur_data = out_data_o + i * FWDCURVES180602_NUMPOINTS;
            auto f_pos = o_base + static_cast<decltype(o_base)>(indList_p[i] * FWDCURVES180602_NUMPOINTS * sizeof(DataType));
            __h_d.clear();
            __h_d.seekg(f_pos, std::ios::beg);
            __h_d.read(reinterpret_cast<char *>(cur_data), FWDCURVES180602_NUMPOINTS * sizeof(DataType));
            if (!__h_d.good())
                throw 0x105;
        }
        npy_intp odims_o[] = { static_cast<npy_intp>(batchNum), FWDCURVES180602_NUMPOINTS };
        PyRes_o = PyArray_SimpleNewFromData(2, odims_o, FWDCURVES180602_DATATYPE, reinterpret_cast<void *>(out_data_o));
        if (PyRes_o == nullptr) {
            throw 0x209;
        }
        PyArray_ENABLEFLAGS((PyArrayObject *)PyRes_o, NPY_ARRAY_OWNDATA);
    }
    catch (int errornum) {
        cerr << cdlu::DLU_error_log(errornum) << endl;
        if (PyRes_i) {
            Py_XDECREF(PyRes_i);
        }
        else if (out_data_i) {
            delete[] out_data_i;
        }
        if (PyRes_o) {
            Py_XDECREF(PyRes_o);
        }
        else if (out_data_o) {
            delete[] out_data_o;
        }
        __h.clear();
        __h_d.clear();
        return nullptr;
    }
    return Py_BuildValue("(OO)", PyRes_i, PyRes_o);
}
bool cdlu::IO_FWM180602::save(string filename) {
    return false;
}
std::ostream & cdlu::IO_FWM180602::__print(std::ostream & out) const {
    auto self_size = size();
    out << "<IOHandle - FWM180602:" << endl;
    auto empty = true;
    if (__h.is_open()) {
        empty = false;
        out << "    |-i:            path=" << __filename << endl;
        out << "    |-i: num. of samples=" << evalSize << endl;
        out << "    |-i:      param dims=" << num_m << endl;
    }
    if (empty) {
        out << "    empty" << endl;
    }
    out << ">";
    return out;
}

cdlu::Projector::Projector(void) :
in_size(0), out_size(0) {
    map_forward.clear();
    map_inverse.clear();
}
cdlu::Projector::~Projector(void) {
    map_forward.clear();
    map_inverse.clear();
}
cdlu::Projector::Projector(const Projector &ref) :
in_size(ref.in_size), out_size(ref.out_size) {
    clear();
    map_forward = ref.map_forward;
    map_inverse = ref.map_inverse;
}
cdlu::Projector& cdlu::Projector::operator=(const Projector &ref) {
    if (this != &ref) {
        clear();
        in_size = ref.in_size;
        out_size = ref.out_size;
        map_forward = ref.map_forward;
        map_inverse = ref.map_inverse;
    }
    return *this;
}
cdlu::Projector::Projector(Projector &&ref) noexcept :
in_size(ref.in_size), out_size(ref.out_size) {
    map_forward = std::move(ref.map_forward);
    map_inverse = std::move(ref.map_inverse);
}
cdlu::Projector& cdlu::Projector::operator=(Projector &&ref) noexcept {
    if (this != &ref) {
        in_size = ref.in_size;
        out_size = ref.out_size;
        map_forward = std::move(ref.map_forward);
        map_inverse = std::move(ref.map_inverse);
    }
    return *this;
}
std::ostream & cdlu::operator<<(std::ostream & out, const cdlu::Projector & self_class) {
    auto self_size = self_class.size();
    out << "<Projector, size=" << self_size << ", i_bdize=" << self_class.in_size << ", o_bdsize=" << self_class.out_size << endl;
    if (self_size > 0) {
        for (auto it_f = self_class.map_forward.begin(); it_f != self_class.map_forward.end(); it_f++) {
            out << "    ( " << it_f->first << " -> " << it_f->second << " )" << endl;
        }
    }
    else {
        out << "    empty" << endl;
    }
    out << ">";
    return out;
}
void cdlu::Projector::clear() {
    in_size = 0;
    out_size = 0;
    map_forward.clear();
    map_inverse.clear();
}
size_t cdlu::Projector::size() const {
    return map_forward.size();
}
bool cdlu::Projector::register_map(PyObject *PyList) {
    if (PyArray_API == nullptr) {
        import_array();
    }
    PyObject *key, *value;
    PyObject *item = 0;
    clear();
    char *datai = nullptr, *datao = nullptr;

    auto it = PyObject_GetIter(PyList);
    if (!it) {
        cerr << DLU_error_log(0x101) << endl;
        return false;
    }
    auto success = true;
    while (item = PyIter_Next(it)) {
        key = PyList_GetItem(item, 0);
        value = PyList_GetItem(item, 1);
        if (PyArray_Check(key) && PyArray_Check(value)) {
            auto bytesize_in = PyArray_NBYTES((PyArrayObject *)key);
            auto bytesize_out = PyArray_NBYTES((PyArrayObject *)value);
            if (in_size == 0 && out_size == 0) {
                in_size  = bytesize_in;
                out_size = bytesize_out;
            }
            else if (bytesize_in != in_size || bytesize_out != out_size) {
                cerr << DLU_error_log(0x002) << "in_size: " << in_size << "?=" << bytesize_in << ", out_size: " << out_size << "?=" << bytesize_out << endl;
                success = false;
                Py_DECREF(item);
                break;
            }
            datai = PyArray_BYTES((PyArrayObject *)key);
            datao = PyArray_BYTES((PyArrayObject *)value);
            map_forward.insert(std::make_pair(DataChunk(datai, bytesize_in), DataChunk(datao, bytesize_out)));
            map_inverse.insert(std::make_pair(DataChunk(datao, bytesize_out), DataChunk(datai, bytesize_in)));
            Py_DECREF(item);
        }
        else {
            cerr << DLU_error_log(0x001) << endl;
            success = false;
            Py_DECREF(item);
            break;
        }
    }
    Py_DECREF(it);
    if (PyErr_Occurred()) {
        cerr << DLU_error_log(0x100) << endl;
    }
    if (!success)
        clear();
    return success;
}
PyObject *cdlu::Projector::__npyintpToPyList(npy_intp *in_list, int size) {
    PyObject *out_list = PyList_New(static_cast<Py_ssize_t>(size));
    for (auto i = 0; i < size; i++) {
        auto e_i = Py_BuildValue("i", in_list[i]);
        PyList_SetItem(out_list, static_cast<Py_ssize_t>(i), e_i);
    }
    return out_list;
}
npy_intp cdlu::Projector::__prodNpy(npy_intp *in_list, int size) {
    npy_intp res = 1;
    for (auto i = 0; i < size; i++) {
        res *= in_list[i];
    }
    return res;
}
PyObject* cdlu::Projector::action(PyObject *PyPicture, bool inversed) {
    if (PyArray_API == nullptr) {
        import_array();
    }
    PyObject *PyResPic = nullptr;
    auto in_size_ref = &in_size;
    auto out_size_ref = &out_size;
    auto map_ref = &map_forward;
    if (inversed) { // re-define the reference.
        in_size_ref = &out_size;
        out_size_ref = &in_size;
        map_ref = &map_inverse;
    }
    try {
        auto in_pic_size = PyArray_NBYTES((PyArrayObject *)PyPicture);
        //cout << "pic_size=" << in_pic_size << endl;
        auto in_data = PyArray_BYTES((PyArrayObject *)PyPicture);
        if (in_data == nullptr) {
            cerr << DLU_error_log(0x003) << endl;
            throw 0x003;
        }
        auto dim_num = PyArray_NDIM((PyArrayObject*)PyPicture);
        auto odims = PyArray_SHAPE((PyArrayObject *)PyPicture);
        auto ochannel = odims[dim_num - 1] * (*out_size_ref);
        auto out_pic_size = in_pic_size / (*in_size_ref) * (*out_size_ref);
        if (ochannel % (*in_size_ref) != 0) {
            throw 0x004;
        }
        else {
            ochannel /= *in_size_ref;
            odims[dim_num - 1] = ochannel;
            //cout << "get inversed ochannel: " << odims[dim_num - 1] << ", in_size_ref=" << *in_size_ref << ", out_size_ref=" << *out_size_ref << endl;
        }
        auto dtype = PyArray_TYPE((PyArrayObject*)PyPicture);
        auto out_data = new char[__prodNpy(odims, dim_num)];

        DataChunk curIndex;
        /*for (auto it_f = map_ref->begin(); it_f != map_ref->end(); it_f++) {
        cout << "    ( " << it_f->first << " -> " << it_f->second << " )" << endl;
        }*/
        //auto out_iter = (PyArrayIterObject *)PyArray_IterNew(PyFrame);
        auto it_o = out_data;
        for (auto it_i = in_data; it_i < in_data + in_pic_size; it_i += (*in_size_ref)) {
            curIndex.set_unsafe(it_i, *in_size_ref);
            //cout << curIndex << "->";
            auto curIt = map_ref->find(curIndex);
            if (curIt != map_ref->end()) {
                //cout << curIt->second << endl;
                it_o = curIt->second.get_copied(it_o);
            }
        }
        //cout << "Complete" << endl;
        curIndex.set_unsafe(nullptr, 0);
        PyResPic = PyArray_SimpleNewFromData(dim_num, odims, dtype, reinterpret_cast<void *>(out_data));
    }
    catch (int errornum) {
        cerr << cdlu::DLU_error_log(errornum) << endl;
        return nullptr;
    }
    return PyResPic;
}

cdlu::DataChunk::DataChunk(void):
size(0), mem(nullptr){
}
cdlu::DataChunk::~DataChunk(void) {
    if (mem) {
        delete[] mem;
    }
}
cdlu::DataChunk::DataChunk(const DataChunk &ref):
size(ref.size), mem(nullptr){
    auto ref_ptr = ref.mem;
    if (size == 0 || (!ref_ptr)) {
        if (mem) {
            delete[] mem;
        }
        mem = nullptr;
        return;
    }
    char *new_ptr = new char[size];
    memcpy(new_ptr, ref_ptr, size);
    mem = new_ptr;
}
cdlu::DataChunk& cdlu::DataChunk::operator=(const DataChunk &ref) {
    if (this != &ref) {
        auto ref_ptr = ref.mem;
        if ((ref.size == 0) || (!ref_ptr)) {
            if (mem) {
                delete[] mem;
            }
            mem = nullptr;
            return *this;
        }
        size = ref.size;
        char *new_ptr = new char[size];
        memcpy(new_ptr, ref_ptr, size);
        mem = new_ptr;
    }
    return *this;
}
cdlu::DataChunk::DataChunk(DataChunk &&ref) noexcept :
size(ref.size), mem(ref.mem){
    ref.mem = nullptr;
}
cdlu::DataChunk& cdlu::DataChunk::operator=(DataChunk &&ref) noexcept{
    if (this != &ref) {
        size = ref.size;
        mem = ref.mem;
        ref.mem = nullptr;
    }
    return *this;
}
cdlu::DataChunk::DataChunk(char *ref, size_t ref_size): DataChunk() {
    if ((ref_size != 0) && (ref != nullptr)) {
        char *new_ptr = new char[ref_size];
        memcpy(new_ptr, ref, ref_size);
        if (mem)
            delete[] mem;
        mem = new_ptr;
        size = ref_size;
    }
}
bool cdlu::DataChunk::operator<(const DataChunk& ref) const {
    return memcmp(mem, ref.mem, size) < 0;
}
bool cdlu::DataChunk::operator==(const DataChunk& ref) const {
    return memcmp(mem, ref.mem, size) == 0;
}
size_t cdlu::DataChunk::hash_func() const {
    size_t __h = 0;
    for (size_t i = 0; i < size; i++)
        __h = 5 * __h + mem[i];
    return __h;
}
std::ostream & cdlu::operator<<(std::ostream & out, const cdlu::DataChunk & self_class) {
    auto p = self_class.mem;
    out << "[" << std::setfill('0') << std::setiosflags(std::ios::uppercase);
    out.setf(std::ios::hex, std::ios::basefield);
    for (auto i = 0; i < self_class.size; i++) {
        out << std::setw(2) << static_cast<int>(static_cast<uint8_t>(*(p+i)));
    }
    out << "]" << std::setfill(' ');
    out.setf(std::ios::dec, std::ios::basefield);
    return out;
}
void cdlu::DataChunk::set(char *ref, size_t ref_size) {
    char *new_ptr = new char[ref_size];
    memcpy(new_ptr, ref, ref_size);
    if (mem)
        delete[] mem;
    mem = new_ptr;
    size = ref_size;
}
void cdlu::DataChunk::set_unsafe(char *ref, size_t ref_size) {
    mem = ref;
    size = ref_size;
}
char* cdlu::DataChunk::get_copied(char *dest) {
    memcpy(dest, mem, size);
    return dest + size;
}
void cdlu::DataChunk::get_copied(char **dataptr) {
    auto out_dataptr = *dataptr;
    memcpy(out_dataptr, mem, size);
}