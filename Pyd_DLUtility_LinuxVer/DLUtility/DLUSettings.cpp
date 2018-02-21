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
    num_shot(0), num_rec(0), num_time(0), IO_Abstract() {
    __filename.clear();
    __folderpath.clear();
}
cdlu::IO_Sesmic::~IO_Sesmic(void) {
}
cdlu::IO_Sesmic::IO_Sesmic(const IO_Sesmic &ref):
    IO_Sesmic() {
    auto refname = ref.__filename;
    if (!refname.empty()) {
        if (!ref.__folderpath.empty()) {
            refname.insert(0, "/");
            refname.insert(0, ref.__folderpath);
        }
        load(refname);
    }
}
cdlu::IO_Sesmic& cdlu::IO_Sesmic::operator=(const IO_Sesmic &ref) {
    if (this != &ref) {
        auto refname = ref.__filename;
        if (!refname.empty()) {
            if (!ref.__folderpath.empty()) {
                refname.insert(0, "/");
                refname.insert(0, ref.__folderpath);
            }
            load(refname);
        }
    }
    return *this;
}
cdlu::IO_Sesmic::IO_Sesmic(IO_Sesmic &&ref) noexcept:
    num_shot(ref.num_shot), num_rec(ref.num_rec), num_time(ref.num_time), IO_Abstract(std::move(ref)) {
    __filename = std::move(ref.__filename);
    __folderpath = std::move(ref.__folderpath);
}
cdlu::IO_Sesmic& cdlu::IO_Sesmic::operator=(IO_Sesmic &&ref) noexcept {
    if (this != &ref) {
        num_shot = ref.num_shot;
        num_rec = ref.num_rec;
        num_time = ref.num_time;
        __filename = std::move(ref.__filename);
        __folderpath = std::move(ref.__folderpath);
        __oh = std::move(ref.__oh);
        __h = std::move(ref.__h);
    }
    return *this;
}
std::ostream & cdlu::operator<<(std::ostream & out, const cdlu::IO_Sesmic & self_class) {
    return self_class.__print(out);
}
void cdlu::IO_Sesmic::clear() {
    IO_Abstract::clear();
    __filename.clear();
    __folderpath.clear();
    num_shot = 0;
    num_rec = 0;
    num_time = 0;
}
PyObject *cdlu::IO_Sesmic::size() const {
    PyObject *pysize = Py_BuildValue("[iii]", num_shot, num_rec, num_time);
    return pysize;
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
    string binname(__filename);
    binname.insert(0, "TDATA_");
    binname.append(".BIN");
    if (!__folderpath.empty()) {
        binname.insert(0, "/");
        binname.insert(0, __folderpath);
    }
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
    if (PyArray_API == nullptr) {
        import_array();
    }
    if (s_num >= num_shot) {
        cerr << cdlu::DLU_error_log(0x007) << ", max position=" << num_shot-1 << endl;
        Py_RETURN_NONE;
    }
    auto thisSize = num_rec * num_time;
    auto offset = s_num * thisSize * sizeof(float);
    __h.clear();
    __h.seekg(offset, std::ios::beg);
    auto out_data = new float[thisSize];
    __h.read(reinterpret_cast<char *>(out_data), thisSize * sizeof(float));
    //size_t get = __h.gcount();
    //printf( "%lld", get);
    //printf("%d - %d\n", ((int*)out_data)[0], ((int*)out_data)[1]);
    npy_intp odims[] = {static_cast<npy_intp>(num_rec), static_cast<npy_intp>(num_time) };
    PyObject *PyResPic = PyArray_SimpleNewFromData(2, odims, NPY_FLOAT32, reinterpret_cast<void *>(out_data));
    return PyResPic;
}
PyObject *cdlu::IO_Sesmic::read(PyObject *s_numPyList) {
    if (PyArray_API == nullptr) {
        import_array();
    }
    auto ochannels = PySequence_Size(s_numPyList);
    if (ochannels == -1) {
        cerr << cdlu::DLU_error_log(0x008) << endl;
        Py_RETURN_NONE;
    }
    auto longcheck = 0;
    for (decltype(ochannels) i = 0; i < ochannels; i++) {
        auto f_obj = PySequence_GetItem(s_numPyList, i);
        if (!f_obj) {
            cerr << cdlu::DLU_error_log(0x009) << endl;
            Py_RETURN_NONE;
        }
        auto f_pos = PyLong_AsLongAndOverflow(f_obj, &longcheck);
        if (longcheck) {
            cerr << cdlu::DLU_error_log(0x009) << endl;
            Py_RETURN_NONE;
        }
        else if (f_pos >= num_shot) {
            cerr << cdlu::DLU_error_log(0x007) << ", occurring at channel " << i << ", max position=" << num_shot - 1 << endl;
            Py_RETURN_NONE;
        }
    }
    auto sliceSize = num_rec * num_time;
    auto dataSize = sliceSize * ochannels;
    auto out_data = new float[dataSize];
    for (decltype(ochannels) i = 0; i < ochannels; i++) {
        auto f_pos = PyLong_AsLong(PySequence_GetItem(s_numPyList, i)) * sliceSize * sizeof(float);
        __h.clear();
        __h.seekg(f_pos, std::ios::beg);
        for (auto *p = out_data+i; p < out_data + dataSize; p += ochannels) {
            __h.read(reinterpret_cast<char *>(p), sizeof(float));
        }
    }
    npy_intp odims[] = { static_cast<npy_intp>(num_rec), static_cast<npy_intp>(num_time), static_cast<npy_intp>(ochannels) };
    PyObject *PyResPic = PyArray_SimpleNewFromData(3, odims, NPY_FLOAT32, reinterpret_cast<void *>(out_data));
    return PyResPic;
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
std::ostream & cdlu::IO_Sesmic::__print(std::ostream & out) const {
    auto self_size = size();
    out << "<IOHandle - Sesmic:" << endl;
    if (__h.is_open()) {
        out << "              folder=" << __folderpath << endl;
        out << "                name=" << __filename << endl;
        out << "       num. of shots=" << num_shot << endl;
        out << "       num. of recv.=" << num_rec << endl;
        out << "      num. of tstep.=" << num_time << endl;
    }
    else {
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
    auto in_size_ref = &in_size;
    auto out_size_ref = &out_size;
    auto map_ref = &map_forward;
    if (inversed) { // re-define the reference.
        in_size_ref = &out_size;
        out_size_ref = &in_size;
        map_ref = &map_inverse;
    }
    auto in_pic_size = PyArray_NBYTES((PyArrayObject *)PyPicture);
    //cout << "pic_size=" << in_pic_size << endl;
    auto in_data = PyArray_BYTES((PyArrayObject *)PyPicture);
    if (in_data == nullptr) {
        cerr << DLU_error_log(0x003) << endl;
        Py_RETURN_NONE;
    }
    auto dim_num = PyArray_NDIM((PyArrayObject*)PyPicture);
    auto odims = PyArray_SHAPE((PyArrayObject *)PyPicture);
    auto ochannel = odims[dim_num - 1] * (*out_size_ref);
    auto out_pic_size = in_pic_size / (*in_size_ref) * (*out_size_ref);
    if (ochannel % (*in_size_ref) != 0) {
        cerr << DLU_error_log(0x004) << endl;
        Py_RETURN_NONE;
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
    PyObject *PyResPic = PyArray_SimpleNewFromData(dim_num, odims, dtype, reinterpret_cast<void *>(out_data));
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