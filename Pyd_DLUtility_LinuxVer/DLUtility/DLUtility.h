#ifndef DLUTILITY_H_INCLUDED
#define DLUTILITY_H_INCLUDED

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <structmember.h>
#include <string>
#include <sstream>

#include "DLUSettings.h"

PyObject *str2PyStr(string Str) {
    //将原始输出转为Unicode
    int wlen = mbstowcs(nullptr, Str.c_str(), 0);
    wchar_t* wszString = new wchar_t[wlen + 1];
    mbstowcs(wszString, Str.c_str(), wlen);
    wszString[wlen] = 0;
    PyObject* res = PyUnicode_FromUnicode((const Py_UNICODE*)wszString, wlen);
    delete[] wszString;
    return res;
}

/*****************************************************************************
* 类/结构的定义:
* 直接引用CMISS_Handle的类
* 除此之外，不在该类中内置任何Python对口的数据，
* 因为相应的数据已经封装在CMISS_Handle中
*****************************************************************************/
// DLU_Projector
typedef struct _C_DLU_Projector
{
    PyObject_HEAD             // == PyObject ob_base;  定义一个PyObject对象.
                              //////////////////////////////////////////////////////////////////////////
                              // 类/结构的真正成员部分.
                              //
    cdlu::Projector *_in_Handle;
}C_DLU_Projector;

static PyMemberDef C_DLU_Proj_DataMembers[] =        //注册类/结构的数据成员.
{ //不注册任何数据，因为类数据CMpegDecoder在上层是不可见的
  //{"m_dEnglish", T_FLOAT,  offsetof(CScore, m_dEnglish), 0, "The English score of instance."},
    { "hAddress",   T_ULONGLONG, offsetof(C_DLU_Projector, _in_Handle),   READONLY, "The address of the handle in memory." },
    { nullptr, 0, 0, 0, nullptr }
};

// DLU_DataIO
typedef struct _C_DLU_DataIO
{
    PyObject_HEAD             // == PyObject ob_base;  定义一个PyObject对象.
                              //////////////////////////////////////////////////////////////////////////
                              // 类/结构的真正成员部分.
                              //
    cdlu::IO_Abstract *_in_Handle;
}C_DLU_DataIO;

static PyMemberDef C_DLU_DtIO_DataMembers[] =        //注册类/结构的数据成员.
{ //不注册任何数据，因为类数据CMpegDecoder在上层是不可见的
  //{"m_dEnglish", T_FLOAT,  offsetof(CScore, m_dEnglish), 0, "The English score of instance."},
    { "hAddress",   T_ULONGLONG, offsetof(C_DLU_DataIO, _in_Handle),   READONLY, "The address of the handle in memory." },
    { nullptr, 0, 0, 0, nullptr }
};

/*****************************************************************************
* 全函数声明:
* 为后续的函数注册准备好声明的对象
*****************************************************************************/

/*static PyObject* C_MPC_Global(PyObject* Self, PyObject *args, PyObject *kwargs) {
    char dumpLevel = -1;
    static char *kwlist[] = { "dumpLevel", nullptr };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|B", kwlist, &dumpLevel)) {
        cerr << "Error.GlobalSettings: invalid keyword'" << endl;
        return nullptr;
    }
    if (dumpLevel != -1) {
        cmpc::__dumpControl = static_cast<int8_t>(dumpLevel);
    }
    Py_RETURN_NONE;
}*/

static PyObject* C_DLU_Help(PyObject* Self) {
    cout << R"(================================================================================
 __, _,    __, _, _ _,_  _, _, _  _, _ _, _  _,   ___  _,  _, _,   _,
 | \ |     |_  |\ | |_| /_\ |\ | / ` | |\ | / _    |  / \ / \ |   (_ 
 |_/ | ,   |   | \| | | | | | \| \ , | | \| \ /    |  \ / \ / | , , )
 ~   ~~~   ~~~ ~  ~ ~ ~ ~ ~ ~  ~  ~  ~ ~  ~  ~     ~   ~   ~  ~~~  ~ 
================================================================================
Yuchen's Deep Learning Enhancing Tools - Readme
    This is a collection of deep learning utilities. You could use it to pre-
        process some data and do something on numpy arrays efficiently.
    For more instructions, you could tap help(dlUtilities). 
================================================================================
V0.7 update report:
    1. Fix some bugs that may cause memory leaking.
    2. Improve the code quality by using try blocks to tackle errors.
    3. Add the mode 'fwm180602' to the 'DataIO.load()'.
    4. Arrange the format of DLUError.
V0.6 update report:
    1. Add the 'save' & 'write' methods for 'DataIO' tool.
V0.55 update report:
    1. Add the 'batchRead' method for 'DataIO' tool.
V0.5 update report:
    1. Provide the 'Projector' tool and 'DataIO' tool.
)";
    Py_RETURN_NONE;
}

/*****************************************************************************
* 重载C_DLU各类的所有内置、构造方法。
*****************************************************************************/

// DLU_Projector
static int C_DLU_Proj_init(C_DLU_Projector* Self, PyObject* args, PyObject *kwargs) { //重载·构造方法.
    PyObject *lmap = nullptr;
    static char *kwlist[] = { "labelMap", nullptr };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "|O", kwlist, &lmap)) {
        PyErr_SetString(PyExc_TypeError, "need 'labelMap(list)'");
        return -1;
    }
    if (lmap) {
        if (!PyList_Check(lmap)) {
            PyErr_SetString(PyExc_TypeError, "need 'labelMap(list)'");
            return -1;
        }
        Self->_in_Handle = new cdlu::Projector;
        Self->_in_Handle->register_map(lmap);
    }
    else {
        Self->_in_Handle = new cdlu::Projector;
    }
    return 0;
}

static void C_DLU_Proj_Destruct(C_DLU_Projector* Self) { //重载·析构方法
    delete Self->_in_Handle;
    Py_TYPE(Self)->tp_free((PyObject*)Self); //释放对象/实例.
}

static PyObject* C_DLU_Proj_Str(C_DLU_Projector* Self) { //重载·调用str/print时自动调用此函数.
    ostringstream OStr;
    OStr << *(Self->_in_Handle);
    return str2PyStr(OStr.str());
}

static PyObject* C_DLU_Proj_Repr(C_DLU_Projector* Self) { //重载·调用repr内置函数时自动调用.
    return C_DLU_Proj_Str(Self);
}

// DLU_DataIO
static int C_DLU_DtIO_init(C_DLU_DataIO* Self) { //重载·构造方法.
    Self->_in_Handle = nullptr;
    return 0;
}

static void C_DLU_DtIO_Destruct(C_DLU_DataIO* Self) { //重载·析构方法
    if (Self->_in_Handle)
        delete Self->_in_Handle;
    Py_TYPE(Self)->tp_free((PyObject*)Self); //释放对象/实例.
}

static PyObject* C_DLU_DtIO_Str(C_DLU_DataIO* Self) { //重载·调用str/print时自动调用此函数.
    if (Self->_in_Handle) {
        ostringstream OStr;
        OStr << *(Self->_in_Handle);
        return str2PyStr(OStr.str());
    }
    else {
        return str2PyStr(string("<IOHandle - Unallocated>"));
    }
}

static PyObject* C_DLU_DtIO_Repr(C_DLU_DataIO* Self) { //重载·调用repr内置函数时自动调用.
    return C_DLU_DtIO_Str(Self);
}

/*****************************************************************************
* 定义C_DLU各类的面向Python接口。
*****************************************************************************/
// Projector
static PyObject* C_DLU_Proj_Register(C_DLU_Projector* Self, PyObject *args, PyObject *kwargs) {
    /* 封装(bool)C_DLU_Proj_Register函数，输入依次为:
    *   labelMap [dict->dict]: 关键字转换字典
    */
    PyObject *lmap = nullptr;
    static char *kwlist[] = { "labelMap", nullptr };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|", kwlist, &lmap)) {
        PyErr_SetString(PyExc_TypeError, "need 'labelMap(list)'");
        return nullptr;
    }
    if (!PyList_Check(lmap)) {
        PyErr_SetString(PyExc_TypeError, "need 'labelMap(list)'");
        return nullptr;
    }
    auto res = Self->_in_Handle->register_map(lmap);
    if (res)
        Py_RETURN_TRUE;
    else
        Py_RETURN_FALSE;
}

static PyObject* C_DLU_Proj_Action(C_DLU_Projector* Self, PyObject *args, PyObject *kwargs) {
    /* 封装(PyArrayObject)C_DLU_Proj_Action函数，输入依次为:
    *   labelMap [dict->dict]: 关键字转换字典
    */
    if (PyArray_API == nullptr) {
        import_array();
    }
    PyObject *inpic = nullptr;
    int inversed = 0;
    static char *kwlist[] = { "PyPicture", "inversed", nullptr };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|p", kwlist, &inpic, &inversed)) {
        PyErr_SetString(PyExc_TypeError, "need 'PyPicture(np.ndarray)', and optional argument 'inversed(bool)'");
        return nullptr;
    }
    if (!PyArray_Check(inpic)) {
        cerr << cdlu::DLU_error_log(0x203) << endl;
        return nullptr;
    }
    bool b_inversed = (inversed != 0);
    auto res = Self->_in_Handle->action(inpic, b_inversed);
    if (res == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, "fail to execute action, please check the DLUError log.");
    }
    return res;
}

static PyObject* C_DLU_Proj_Clear(C_DLU_Projector* Self) {
    /* 封装(void)clear函数，输入必须空置 */
    Self->_in_Handle->clear();
    Py_RETURN_NONE;
}

// DataIO
static PyObject* C_DLU_DtIO_Clear(C_DLU_DataIO* Self) {
    /* 封装(void)clear函数，输入必须空置 */
    if (Self->_in_Handle) {
        Self->_in_Handle->clear();
        delete Self->_in_Handle;
        Self->_in_Handle = nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* C_DLU_DtIO_Close(C_DLU_DataIO* Self) {
    /* 封装(void)close函数，输入必须空置 */
    if (Self->_in_Handle) {
        Self->_in_Handle->close();
        delete Self->_in_Handle;
        Self->_in_Handle = nullptr;
    }
    else {
        PyErr_SetString(PyExc_IOError, "Should not close a file without loading/saving it.");
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* C_DLU_DtIO_Size(C_DLU_DataIO* Self) {
    /* 封装(?)size函数，输入必须空置 */
    if (Self->_in_Handle) {
        auto sizeobj = Self->_in_Handle->size();
        if (!sizeobj) {
            PyErr_SetString(PyExc_NotImplementedError, "The current mode does not provide this function.");
            return nullptr;
        }
        else {
            return sizeobj;
        }
    }
    else {
        PyErr_SetString(PyExc_IOError, "Should not detect size without loading/saving file.");
        return nullptr;
    }
    Py_RETURN_NONE;
}

static PyObject* C_DLU_DtIO_Load(C_DLU_DataIO* Self, PyObject *args, PyObject *kwargs) {
    /* 封装(bool)load函数，输入依次为:
    *   filepath [bytes->string]: 文件路径
    *   mode     [bytes->string]: 工作模式
    */
    if (Self->_in_Handle) {
        PyErr_SetString(PyExc_IOError, "Should close/clear the IO handle before loading a new file.");
        return nullptr;
    }
    Py_buffer py_fpath = { 0 };
    Py_buffer py_mode = { 0 };
    static char *kwlist[] = { "filePath", "mode", nullptr };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*|y*", kwlist, &py_fpath, &py_mode)) {
        PyErr_SetString(PyExc_TypeError, "need 'filePath(bytes)', and optional argument 'mode(bytes)'");
        return nullptr;
    }
    string in_fpath;
    if (py_fpath.buf)
        in_fpath.assign(reinterpret_cast<char *>(py_fpath.buf));
    else {
        PyErr_SetString(PyExc_TypeError, "Must specify a valid path.");
        if (py_mode.buf)
            PyBuffer_Release(&py_mode);
        return nullptr;
    }
    PyBuffer_Release(&py_fpath);
    string in_mode;
    if (py_mode.buf)
        in_mode.assign(reinterpret_cast<char *>(py_mode.buf));
    else
        in_mode.assign("seismic");
    PyBuffer_Release(&py_mode);
    if (in_mode.compare("seismic") == 0) {
        Self->_in_Handle = new cdlu::IO_Sesmic;
    }
    else if (in_mode.compare("fwm180602") == 0) {
        Self->_in_Handle = new cdlu::IO_FWM180602;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "The assigned mode is not valid.");
        return nullptr;
    }
    auto success = Self->_in_Handle->load(in_fpath);
    if (success) {
        Py_RETURN_TRUE;
    }
    else {
        PyErr_SetString(PyExc_IOError, "Unable to load the specified file, please check the DLU-Error for details.");
        delete Self->_in_Handle;
        Self->_in_Handle = nullptr;
        return nullptr;
    }
}

static PyObject* C_DLU_DtIO_Save(C_DLU_DataIO* Self, PyObject *args, PyObject *kwargs) {
    /* 封装(bool)save函数，输入依次为:
    *   filepath [bytes->string]: 文件路径
    *   mode     [bytes->string]: 工作模式
    */
    if (Self->_in_Handle) {
        PyErr_SetString(PyExc_IOError, "Should close/clear the IO handle before saving a new file.");
        return nullptr;
    }
    Py_buffer py_fpath = { 0 };
    Py_buffer py_mode = { 0 };
    static char *kwlist[] = { "filePath", "mode", nullptr };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "y*|y*", kwlist, &py_fpath, &py_mode)) {
        PyErr_SetString(PyExc_TypeError, "need 'filePath(bytes)', and optional argument 'mode(bytes)'");
        return nullptr;
    }
    string in_fpath;
    if (py_fpath.buf)
        in_fpath.assign(reinterpret_cast<char *>(py_fpath.buf));
    else {
        PyErr_SetString(PyExc_TypeError, "Must specify a valid path.");
        if (py_mode.buf)
            PyBuffer_Release(&py_mode);
        return nullptr;
    }
    PyBuffer_Release(&py_fpath);
    string in_mode;
    if (py_mode.buf)
        in_mode.assign(reinterpret_cast<char *>(py_mode.buf));
    else
        in_mode.assign("seismic");
    PyBuffer_Release(&py_mode);
    if (in_mode.compare("seismic") == 0) {
        Self->_in_Handle = new cdlu::IO_Sesmic;
    }
    else {
        PyErr_SetString(PyExc_TypeError, "The assigned mode is not valid.");
        return nullptr;
    }
    auto success = Self->_in_Handle->save(in_fpath);
    if (success) {
        Py_RETURN_TRUE;
    }
    else {
        PyErr_SetString(PyExc_IOError, "Unable to save the specified file, please check the DLU-Error for details.");
        delete Self->_in_Handle;
        Self->_in_Handle = nullptr;
        return nullptr;
    }
}

static PyObject* C_DLU_DtIO_Read(C_DLU_DataIO* Self, PyObject *args, PyObject *kwargs) {
    /* 封装(ndarray)read函数，输入依次为:
    *   indices [int/tuple]: 下标/多下标
    */
    if (!Self->_in_Handle) {
        PyErr_SetString(PyExc_IOError, "Should not read without loading file.");
        return nullptr;
    }
    PyObject *indices = nullptr;
    static char *kwlist[] = { "indicies", nullptr };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|", kwlist, &indices)) {
        PyErr_SetString(PyExc_TypeError, "need 'indices(int/sequence)'");
        return nullptr;
    }
    if (PyLong_Check(indices)) {
        auto index = PyLong_AsLong(indices);
        auto res = Self->_in_Handle->read(index);
        if (!res) {
            PyErr_SetString(PyExc_NotImplementedError, "Meet a fatal error, or the current mode does not provide this function.");
            return nullptr;
        }
        else {
            return res;
        }
    }
    else {
        auto res = Self->_in_Handle->read(indices);
        if (!res) {
            PyErr_SetString(PyExc_NotImplementedError, "Meet a fatal error, or the current mode does not provide this function.");
            return nullptr;
        }
        else {
            return res;
        }
    }
    Py_RETURN_NONE;
}

static PyObject* C_DLU_DtIO_BatchRead(C_DLU_DataIO* Self, PyObject *args, PyObject *kwargs) {
    /* 封装(ndarray)batchread函数，输入依次为:
    *   batchnum     [int]: batch数目
    *   shape      [tuple]: 尺寸
    */
    if (!Self->_in_Handle) {
        PyErr_SetString(PyExc_IOError, "Should not read without loading file.");
        return nullptr;
    }
    int batchnum = 0;
    PyObject *shape = nullptr;
    static char *kwlist[] = { "batchNum", "shape", nullptr };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "iO|", kwlist, &batchnum, &shape)) {
        PyErr_SetString(PyExc_TypeError, "need 'batchNum(int)', 'shape(sequence)' and an optional 'multithread(bool)'");
        return nullptr;
    }
    auto res = Self->_in_Handle->read(batchnum, shape);
    if (!res) {
        PyErr_SetString(PyExc_NotImplementedError, "Meet a fatal error, or the current mode does not provide this function.");
        return nullptr;
    }
    else {
        return res;
    }
    Py_RETURN_NONE;
}

static PyObject* C_DLU_DtIO_Write(C_DLU_DataIO* Self, PyObject *args, PyObject *kwargs) {
    /* 封装(size_t)write函数，输入依次为:
    *   indices [int/tuple]: 下标/多下标
    */
    if (PyArray_API == nullptr) {
        import_array();
    }
    if (!Self->_in_Handle) {
        PyErr_SetString(PyExc_IOError, "Should not write without saving file.");
        return nullptr;
    }
    PyObject *data = nullptr;
    static char *kwlist[] = { "data", nullptr };
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O|", kwlist, &data)) {
        PyErr_SetString(PyExc_TypeError, "need 'data(ndarray)'");
        return nullptr;
    }
    if (PyArray_Check(data)) {
        auto res = Self->_in_Handle->write(data);
        if (res == -1) {
            PyErr_SetString(PyExc_NotImplementedError, "Meet errors while writing, or the current mode does not provide this function.");
            return nullptr;
        }
        else {
            return Py_BuildValue("i", res);
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "need 'data(ndarray)'");
        return nullptr;
    }
    Py_RETURN_NONE;
}

//注意下面这俩函数，为什么它们不需要Py_IN/DECREF呢？因为未创建临时变量，也没有
//使用形如None这样的现成返回对象！
/*static PyObject* FreePyArray(PyArrayObject *PyArray) {
    if (PyArray_API == nullptr) {
        import_array();
    }
    uint8_t * out_dataptr = (uint8_t *)PyArray_DATA(PyArray);
    delete [] out_dataptr;
    return nullptr;
}
void FreePyList(PyObject *PyList) {
    Py_ssize_t getlen = PyList_Size(PyList);
    for (Py_ssize_t i = 0; i < getlen; i++) {
        PyObject *Item = PyList_GetItem(PyList, i);
        FreePyArray((PyArrayObject*)Item);
    }
    Py_DECREF(PyList);
    int x = PyList_ClearFreeList();
    //cout << "Freed: " << x << " items" << endl;
}*/

/*****************************************************************************
* 函数模块登录与注册
*****************************************************************************/
static PyMethodDef C_DLU_MethodMembers[] =      //注册全局函数列表
{
    { "readme",          (PyCFunction)C_DLU_Help,               METH_NOARGS, \
    "Use it to see readme and some useful instructions." },
    { nullptr, nullptr, 0, nullptr }
};

static PyMethodDef C_DLU_Proj_MethodMembers[] =      //注册类的所有成员函数结构列表.
{ //该步的意义即为进一步封装CMISS_Handle，为其提供面向Python的接口
    { "clear",              (PyCFunction)C_DLU_Proj_Clear,                METH_NOARGS, \
    "clear the settings of this projector." },
    { "registerMap",        (PyCFunction)C_DLU_Proj_Register,             METH_VARARGS | METH_KEYWORDS, \
    "Use a numpy-based dict to setup this projector.\n - labelMap: [list] an n*2 list which defines the axes of projecting method." },
    { "action",             (PyCFunction)C_DLU_Proj_Action,               METH_VARARGS | METH_KEYWORDS, \
    "Project the input ndarray into another axis space.\n - PyPicture: [ndarray] the input picture which needs to be projected.\n - inversed: [bool](o) whether the projecting direction is inversed." },
    { nullptr, nullptr, 0, nullptr }
};

static PyMethodDef C_DLU_DtIO_MethodMembers[] =      //注册类的所有成员函数结构列表.
{
    { "clear",              (PyCFunction)C_DLU_DtIO_Clear,                METH_NOARGS, \
    "clear the settings of this IO Handle, always avaliable." },
    { "close",              (PyCFunction)C_DLU_DtIO_Close,                METH_NOARGS, \
    "close a loaded/saved IO handle, avaliable only after we have called load/save method." },
    { "size",               (PyCFunction)C_DLU_DtIO_Size,                 METH_NOARGS, \
    "Return the size of file in current handle. Sometimes it has other meanings and sometimes this\n method is not avaliable. It depends on the mode where the handle works." },
    { "load",               (PyCFunction)C_DLU_DtIO_Load,                 METH_VARARGS | METH_KEYWORDS, \
    "Load a resource file.\n - filePath: [bytes] a path which defines where the file stored.\n - mode: [bytes] the assigned mode of the IN handle (default: 'seismic')." },
    { "save",               (PyCFunction)C_DLU_DtIO_Save,                 METH_VARARGS | METH_KEYWORDS, \
    "Create a destination file for saving data.\n - filePath: [bytes] a path which defines where the file stored.\n - mode: [bytes] the assigned mode of the OUT handle (default: 'seismic')." },
    { "read",               (PyCFunction)C_DLU_DtIO_Read,                 METH_VARARGS | METH_KEYWORDS, \
    "Read a data chunk.\n - indices: [int/sequence] the indices (or index) of fetched data." },
    { "write",              (PyCFunction)C_DLU_DtIO_Write,                METH_VARARGS | METH_KEYWORDS, \
    "Write an array as a data chunk to the destination.\n - data: [ndarray] the data that needs to be written." },
    { "batchRead",          (PyCFunction)C_DLU_DtIO_BatchRead,            METH_VARARGS | METH_KEYWORDS, \
    "Use a random strategy to read a data batch.\n - batchNum: [int] the number of fetched sample data.\n - shape: [sequence] the shape of returned data." },
    { nullptr, nullptr, 0, nullptr }
};

/*****************************************************************************
* 类/结构的所有成员、内置属性的说明信息..
* 为Python类提供顶层的封装
*****************************************************************************/
static PyTypeObject C_DLU_Proj_ClassInfo =
{
    PyVarObject_HEAD_INIT(nullptr, 0)"dlUtilities.Projector",  //可以通过__class__获得这个字符串. CPP可以用类.__name__获取.
    sizeof(C_DLU_Projector),                 //类/结构的长度.调用PyObject_New时需要知道其大小.
    0,
    (destructor)C_DLU_Proj_Destruct,    //类的析构函数.
    0,
    0,
    0,
    0,
    (reprfunc)C_DLU_Proj_Repr,
    0,
    0,
    0,
    0,
    0,
    (reprfunc)C_DLU_Proj_Str,         //Str/print内置函数调用.
    0,
    0,
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     //如果没有提供方法的话，为Py_TPFLAGS_DEFAULE
    "This class is used to project a numpy array to another form based on a key map.",   //__doc__,类/结构的DocString.
    0,
    0,
    0,
    0,
    0,
    0,
    C_DLU_Proj_MethodMembers,       //类的所有方法集合.
    C_DLU_Proj_DataMembers,         //类的所有数据成员集合.
    0,
    0,
    0,
    0,
    0,
    0,
    (initproc)C_DLU_Proj_init,      //类的构造函数.
    0,
};

static PyTypeObject C_DLU_DtIO_ClassInfo =
{
    PyVarObject_HEAD_INIT(nullptr, 0)"dlUtilities.DataIO",  //可以通过__class__获得这个字符串. CPP可以用类.__name__获取.
    sizeof(C_DLU_DataIO),                 //类/结构的长度.调用PyObject_New时需要知道其大小.
    0,
    (destructor)C_DLU_DtIO_Destruct,    //类的析构函数.
    0,
    0,
    0,
    0,
    (reprfunc)C_DLU_DtIO_Repr,
    0,
    0,
    0,
    0,
    0,
    (reprfunc)C_DLU_DtIO_Str,         //Str/print内置函数调用.
    0,
    0,
    0,
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,     //如果没有提供方法的话，为Py_TPFLAGS_DEFAULE
    "This class is used as a high-level API for low-level fast reading/writing data.\nNoted that not always is every method provided. It depends on the mode where you let this instance work.",   //__doc__,类/结构的DocString.
    0,
    0,
    0,
    0,
    0,
    0,
    C_DLU_DtIO_MethodMembers,       //类的所有方法集合.
    C_DLU_DtIO_DataMembers,         //类的所有数据成员集合.
    0,
    0,
    0,
    0,
    0,
    0,
    (initproc)C_DLU_DtIO_init,      //类的构造函数.
    0,
};

/*****************************************************************************
* 此模块说明信息..
* 为Python模块提供顶层的封装
*****************************************************************************/
static PyModuleDef ModuleInfo =
{
    PyModuleDef_HEAD_INIT,
    "dlUtilities",               //模块的内置名--__name__.
    "A collection of deep learning enhancing tools.",  //模块的DocString.__doc__
    -1,
    nullptr, nullptr, nullptr, nullptr, nullptr
};

#endif