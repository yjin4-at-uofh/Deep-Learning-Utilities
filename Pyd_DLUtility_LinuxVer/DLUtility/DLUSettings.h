#ifndef DLUSETTINGS_H_INCLUDED
#define DLUSETTINGS_H_INCLUDED

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define DLUSETTINGS_CURRENT_VERSION "0.5"

#include <unordered_map>
#include <fstream>
#include <sstream>
#include <Python.h>
#include <numpy/arrayobject.h>

#include "DLUErrorLog.h"

#define max(a,b) ((a) > (b) ? (a) : (b))

namespace cdlu {

    class DataChunk {
    public:
        DataChunk(void); //三五法则，定义其中之一就必须全部手动定义
        ~DataChunk(void);
        DataChunk(const DataChunk &ref);
        DataChunk& operator=(const DataChunk &ref);
        DataChunk(DataChunk &&ref) noexcept;
        DataChunk& operator=(DataChunk &&ref) noexcept;
        DataChunk(char *ref, size_t ref_size);
        // 运算符重载
        bool operator<(const DataChunk& ref) const; // 比较运算符，用于构建字典
        bool operator==(const DataChunk& ref) const;
        size_t hash_func() const;
        friend std::ostream & operator<<(std::ostream & out, const DataChunk & self_class); // 用于显示
        // 功能部分
        void set(char *ref, size_t ref_size); // 从一个数组置入数据
        void set_unsafe(char *ref, size_t ref_size); // 从一个数组用移动赋值法置入数据，注意这是一种不保险的做法
        char* get_copied(char *dest); // 置入操作符，将自己的内存段放入一个数组，并返回位移后的数组指针
        void get_copied(char **dataptr); // 置入操作符，为Numpy_Iter设计的版本
        size_t size;
    private:
        char *mem;
        
    };

    class Projector{ /*高效率通道投影装置*/
    public:
        Projector(void);                                      //构造函数
        // 以下部分就是传说中的三五法则，定义其中之一就必须全部手动定义
        ~Projector(void);                                     //析构函数
        Projector(const Projector &ref);                      //拷贝构造函数
        Projector& operator=(const Projector &ref);           //拷贝赋值函数
        Projector(Projector &&ref) noexcept;                  //移动构造函数
        Projector& operator=(Projector &&ref) noexcept;       //移动赋值函数
        // 运算符重载
        friend std::ostream & operator<<(std::ostream & out, const Projector & self_class); // 用于显示
        void clear();
        size_t size() const;
        bool register_map(PyObject *PyList);
        PyObject* action(PyObject *PyPicture, bool inversed=false);
    private:
        size_t in_size;
        size_t out_size;
        struct _cmp_hash {
            size_t operator()(const DataChunk &a) const {
                return a.hash_func();
            }
        };
        struct _cmp_eq {
            bool operator()(const DataChunk &a, const DataChunk &b) const {
                return a == b;
            }
        };
        std::unordered_map<DataChunk, DataChunk, _cmp_hash, _cmp_eq> map_forward;
        std::unordered_map<DataChunk, DataChunk, _cmp_hash, _cmp_eq> map_inverse;
        PyObject *__npyintpToPyList(npy_intp *in_list, int size);
        npy_intp __prodNpy(npy_intp *in_list, int size);
    };

    class IO_Abstract {
    public:
        IO_Abstract(void);                                        //构造函数
        // 以下部分就是传说中的三五法则，定义其中之一就必须全部手动定义
        virtual ~IO_Abstract(void);                               //析构函数
        IO_Abstract(const IO_Abstract &ref) = delete;             //禁用拷贝构造函数
        IO_Abstract& operator=(const IO_Abstract &ref) = delete;  //禁用拷贝赋值函数
        IO_Abstract(IO_Abstract &&ref) noexcept;                  //移动构造函数
        IO_Abstract& operator=(IO_Abstract &&ref) noexcept;       //移动赋值函数
        // 运算符重载
        friend std::ostream & operator<<(std::ostream & out, const IO_Abstract & self_class); // 用于显示
        virtual void clear();
        virtual PyObject *size() const;
        virtual bool load(string filename) = 0;
        virtual void close() = 0;
        virtual PyObject *read(size_t s_num);
        virtual PyObject *read(PyObject *s_numPyList);
    protected:
        std::ifstream __h;
        std::ofstream __oh;
        // 协同重载运算符的虚函数
        virtual std::ostream & __print(std::ostream & out) const;
    };

    class IO_Sesmic : public IO_Abstract { /**/
    public:
        IO_Sesmic(void);                                      //构造函数
                                                              // 以下部分就是传说中的三五法则，定义其中之一就必须全部手动定义
        ~IO_Sesmic(void);                                     //析构函数
        IO_Sesmic(const IO_Sesmic &ref);                      //拷贝构造函数
        IO_Sesmic& operator=(const IO_Sesmic &ref);           //拷贝赋值函数
        IO_Sesmic(IO_Sesmic &&ref) noexcept;                  //移动构造函数
        IO_Sesmic& operator=(IO_Sesmic &&ref) noexcept;       //移动赋值函数
        // 运算符重载
        friend std::ostream & operator<<(std::ostream & out, const IO_Sesmic & self_class); // 用于显示
        void clear() override;
        PyObject *size() const override;
        bool load(string filename) override;
        void close() override;
        PyObject *read(size_t s_num) override;
        PyObject *read(PyObject *s_numPyList) override;
    private:
        string __filename; //用来拷贝构造和赋值的临时变量
        string __folderpath;
        size_t num_shot;
        size_t num_rec;
        size_t num_time;
        bool __read_log_info();
    protected:
        std::ostream & __print(std::ostream & out) const override;
    };
}

#endif