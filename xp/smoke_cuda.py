import ctypes as c
import os
import time
import uuid

from hardware import processes


def compute(delay=0):
    selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not selected.startswith('GPU-') or ',' in selected:
        raise RuntimeError('CUDA smoke requires one explicit GPU UUID')
    if processes().get(selected,set())-{os.getpid()}:
        raise RuntimeError('CUDA smoke GPU has a foreign compute process')
    lib=c.CDLL('libcuda.so.1')
    def call(name,types,*args):
        fn=getattr(lib,name)
        fn.argtypes=types
        fn.restype=c.c_int
        result=fn(*args)
        if result:
            raise RuntimeError(f'{name} failed with CUDA code {result}')
    call('cuInit',[c.c_uint],0)
    count=c.c_int()
    call('cuDeviceGetCount',[c.POINTER(c.c_int)],c.byref(count))
    device=None
    for index in range(count.value):
        ident=(c.c_ubyte*16)()
        call('cuDeviceGetUuid',[c.c_void_p,c.c_int],ident,index)
        if 'GPU-'+str(uuid.UUID(bytes=bytes(ident)))==selected:
            device=index
            break
    if device is None:
        raise RuntimeError('selected GPU UUID not visible to CUDA driver')
    context=c.c_void_p()
    call('cuCtxCreate_v2',[c.POINTER(c.c_void_p),c.c_uint,c.c_int],c.byref(context),0,device)
    pointer=c.c_uint64()
    module=c.c_void_p()
    try:
        host=(c.c_float*32)(*range(32))
        call('cuMemAlloc_v2',[c.POINTER(c.c_uint64),c.c_size_t],c.byref(pointer),c.sizeof(host))
        call('cuMemcpyHtoD_v2',[c.c_uint64,c.c_void_p,c.c_size_t],pointer,host,c.sizeof(host))
        ptx=c.create_string_buffer(b'''
.version 7.0
.target sm_75
.address_size 64
.visible .entry increment(.param .u64 data) {
    .reg .b32 r;
    .reg .b64 p, offset;
    .reg .f32 value;
    ld.param.u64 p, [data];
    mov.u32 r, %tid.x;
    mul.wide.u32 offset, r, 4;
    add.u64 p, p, offset;
    ld.global.f32 value, [p];
    add.f32 value, value, 0f3F800000;
    st.global.f32 [p], value;
    ret;
}
''')
        call('cuModuleLoadData',[c.POINTER(c.c_void_p),c.c_void_p],c.byref(module),ptx)
        fn=c.c_void_p()
        call('cuModuleGetFunction',[c.POINTER(c.c_void_p),c.c_void_p,c.c_char_p],c.byref(fn),module,b'increment')
        args=(c.c_void_p*1)(c.cast(c.byref(pointer),c.c_void_p))
        call('cuLaunchKernel',[c.c_void_p]+[c.c_uint]*7+[c.c_void_p]*3,
             fn,1,1,1,32,1,1,0,None,args,None)
        call('cuCtxSynchronize',[])
        call('cuMemcpyDtoH_v2',[c.c_void_p,c.c_uint64,c.c_size_t],host,pointer,c.sizeof(host))
        if list(host)!=[float(i+1) for i in range(32)]:
            raise RuntimeError('tiny fp32 kernel produced incorrect output')
        time.sleep(delay)
        return list(host)
    finally:
        if module: call('cuModuleUnload',[c.c_void_p],module)
        if pointer.value: call('cuMemFree_v2',[c.c_uint64],pointer)
        call('cuCtxDestroy_v2',[c.c_void_p],context)
