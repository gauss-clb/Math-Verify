# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import time
from datetime import datetime
import logging
import os
import json
import multiprocessing
import functools
import dill
import queue
import hashlib
from multiprocessing import Manager, shared_memory, Pool, TimeoutError
import signal
import re
from sympy import evaluate

from math_verify.errors import TimeoutException
from math_verify.constants import _SHARED_CACHE, _CACHE_LOCK

TIMEOUT_WARNING_SHOWN = False
logger = logging.getLogger(__name__)

def write_jsonl(items, path, mode='w'):
    with open(path, mode, encoding='utf8') as fw:
        for item in items:
            fw.write(json.dumps(item, ensure_ascii=False) + '\n')


def write_text(text, path, mode='w'):
    with open(path, mode, encoding='utf8') as fw:
        fw.write(text + '\n')

def load_expr(expr_bytes):
    with evaluate(False): # 防止表示式计算，出现超级大整数超时
        return dill.loads(expr_bytes)  # 反序列化

def get_cache_key(*args, **kwargs):
    # print(f'==== args: {args}, kwargs: {kwargs}')
    # 将 args 和 kwargs 序列化成 str，并计算 hash
    dumped_args = []
    for arg in args:
        if isinstance(arg, re.Match):
            dumped_args.append({
                'groupdict': sorted(arg.groupdict().items()),
                'groups': arg.groups(),
                # 'start': arg.start(),
                # 'end': arg.end(),
                # 'span': arg.span(),
            })
            # print('*********************************')
            # print(arg.groupdict(), arg.groups())
            # print('*********************************')
        else:
            dumped_args.append(arg)
    dumped_kwargs = sorted(kwargs.items())
    # keep dict order
    args_str = dill.dumps(dumped_args)
    kwargs_str = dill.dumps(dumped_kwargs)
    # cache_str = f"{args_str}:{kwargs_str}"
    # print(cache_str)
    # return hash(cache_str)  # 或者直接返回 cache_str 作为键
    # print(hashlib.md5(args_str+kwargs_str).hexdigest())
    return hashlib.md5(args_str+kwargs_str).hexdigest()


def cache_decorator(func):
    cache = {}
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # 生成唯一的缓存键（确保参数可哈希）
        cache_key = get_cache_key(func.__name__, args, kwargs)
        # print([args])
        # print(cache_key)

        if cache_key in cache: # 命中缓存
            return cache[cache_key]
        result = func(*args, **kwargs)
        cache[cache_key] = result
        return result
    return wrapper
        

save_path = '/cpfs/align/chenliangbo/models/verl_checkpoints/ds7b_sys_dspposreuse_timeout/memory.txt'
# save_path2 = '/cpfs/align/chenliangbo/models/verl_checkpoints/ds7b_sys_dspnegreuse_timeout/subprocess_memory3.txt'
# timeout = func_set_timeout

'''
def timeout(timeout_seconds: int | None = 10, use_cache: bool = False):  # noqa: C901
    """A decorator that applies a timeout to the decorated function.

    Args:
        timeout_seconds (int): Number of seconds before timing out the decorated function.
            Defaults to 10 seconds.

    Notes:
        On Unix systems, uses a signal-based alarm approach which is more efficient as it doesn't require spawning a new process.
        On Windows systems, uses a multiprocessing-based approach since signal.alarm is not available. This will incur a huge performance penalty.
    """
    if timeout_seconds is None or timeout_seconds <= 0:

        def no_timeout_decorator(func):
            return func

        return no_timeout_decorator

    if os.name == "posix":
        # Unix-like approach: signal.alarm
        import signal

        def decorator(func):
            def handler(signum, frame):
                raise TimeoutException("Operation timed out!")

            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    # from line_profiler import LineProfiler
                    # from math_verify.parser import extract_match
                    # profiler = LineProfiler()
                    # profiler.add_function(extract_match)
                    # profiler_func = profiler(func)
                    # result = profiler_func(*args, **kwargs)
                    # profiler.print_stats()
                    # return result
                    s = time.time()
                    result = func(*args, **kwargs)
                    e = time.time() - s
                    logger.error(f'{func.__name__}: {e} ...')
                    # # write_jsonl([{'func': func.__name__, 'time': e}], '/cpfs/align/chenliangbo/workspace/test/his.jsonl', 'a')
                    # time.sleep(0.1)
                    return result
                finally:
                    # Cancel the alarm and restore previous handler
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)

            return wrapper

        return decorator

    else:
        # Windows approach: use multiprocessing
        from multiprocessing import Process, Queue

        def decorator(func):
            def wrapper(*args, **kwargs):
                q = Queue()

                def run_func(q, args, kwargs):
                    try:
                        result = func(*args, **kwargs)
                        q.put((True, result))
                    except Exception as e:
                        q.put((False, e))

                p = Process(target=run_func, args=(q, args, kwargs))
                p.start()
                p.join(timeout_seconds)

                if p.is_alive():
                    # Timeout: Terminate the process
                    p.terminate()
                    p.join()
                    raise TimeoutException("Operation timed out!")

                # If we got here, the process completed in time.
                success, value = q.get()
                if success:
                    return value
                else:
                    # The child raised an exception; re-raise it here
                    raise value

            return wrapper

        return decorator
'''

def timeout(timeout_seconds: int = 10, use_cache: bool = False):
    if use_cache:
        # 创建共享缓存（跨进程可用）
        manager = Manager()
        cache = manager.dict()  # 多进程安全的字典

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if use_cache:
                # 生成唯一的缓存键（确保参数可哈希）
                cache_key = get_cache_key(args, kwargs)
                # print(cache_key)

                # 检查缓存
                if cache_key in cache:
                    return load_expr(cache[cache_key])

            def worker(result_queue, *args, **kwargs):
                # write_text(f'{func.__name__}[sub]: enter1', save_path2, 'a')
                # s = time.time()
                try:
                    # from line_profiler import LineProfiler
                    # from math_verify.parser import extract_match
                    # profiler = LineProfiler()
                    # profiler.add_function(extract_match)
                    # profiler_func = profiler(func)
                    # result = profiler_func(*args, **kwargs)
                    # profiler.print_stats()
                    # s = time.time()
                    result = func(*args, **kwargs)
                    # e = time.time() - s
                    # logger.error(f'{func.__name__}: {e} ...')
                    # if func.__name__ == 'extract_target_from_pred':
                    #     write_jsonl([{'func': func.__name__, 'time': e}], '/cpfs/align/chenliangbo/workspace/test/train_cache.jsonl', 'a')
                    # time.sleep(0.1)
                    # s = time.time()
                    if isinstance(result, list):
                        for elem in result:
                            if isinstance(elem, (str, int, float, bool)):
                                result_queue.put(dill.dumps(elem))
                        if result_queue.empty():
                            time.sleep(0.000001)
                        for elem in result:
                            if not isinstance(elem, (str, int, float, bool)):
                                elem_dumped = dill.dumps(elem)
                                # print(type(elem), len(elem_dumped))
                                load_expr(elem_dumped) # maybe timeout, enter infinite recursion
                                result_queue.put_nowait(elem_dumped)
                        if result_queue.empty():
                            time.sleep(0.000001)
                    elif isinstance(result, (str, int, float, bool)):
                        result_queue.put_nowait(dill.dumps(result))
                        if result_queue.empty():
                            time.sleep(0.000001)
                    else:
                        raise Exception('Type is not supported!')
                    # logger.error(f'====================: {time.time() - s} ...')
                        
                except Exception as e:
                    logger.exception(f'Consume time[put error]')
                

            start = time.time()
            result_queue = multiprocessing.Queue(maxsize=10)
            p = multiprocessing.Process(target=worker, args=(result_queue, *args), kwargs=kwargs)
            # logger.error(f'?????????????????: {time.time() - start} ...')
            # start = time.time()
            p.start()
            # logger.error(f'?????????????????: {time.time() - start} ...')
            # start = time.time()
            p.join(timeout=timeout_seconds)
            # logger.error(f'++++++++++++++++++++: {time.time() - start} ...')

            # print('5555555555555')

            # kwargs['log_path'] = '/cpfs/align/chenliangbo/models/verl_checkpoints/ds7b_sys_dspnegreuse_timeout/log3.jsonl'
            # write_text(f'{func.__name__}[main]: enter', save_path, 'a')
            # print('=====')
            # time.sleep(0.1)
            # print(result_queue.get())
            is_timeout = False
            if p.is_alive():
                # write_text(f'{func.__name__}[main1]: enter', save_path, 'a')
                is_timeout = True
                p.terminate()
                p.join(timeout=1.0)
                if p.is_alive():
                    # write_text(f'进程无法终止！kill。', save_path, 'a')
                    p.kill()  # 强制终止
                    p.join(timeout=0.5)  # 最后清理
                    # if p.is_alive():  # 仍然存活（几乎不可能）
                        # write_text(f'进程无法终止！可能是内核问题。', save_path, 'a')
                        # raise Exception("进程无法终止！可能是内核问题。")
                    
                log_path = kwargs.get('log_path', None)
                logger.error('&&&&&&&&&&&&&&&&&&&&&&&&&')
                if log_path:
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if func.__name__ == 'extract_target_from_pred':
                        write_jsonl([{'time': time_now, 'first_expr': str(args[0]), 'status': 'timeout'}], log_path, 'a')
                    else:
                        write_jsonl([{'time': time_now, 'first_expr': str(args[0]), 'second_expr': str(args[1]), 'status': 'timeout'}], log_path, 'a')
                logger.error(f'Consume time[timeout]: {time.time() - start} ...')
                # raise TimeoutException("Operation timed out!")
            
            # write_text(f'{func.__name__}[main2]: enter', save_path, 'a')
            # print('7777777777777777')
            if result_queue.empty():
                # write_text(f'{func.__name__}[main_empty]: enter', save_path, 'a')
                log_path = kwargs.get('log_path', None)
                logger.error('#######################')
                if log_path:
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if func.__name__ == 'extract_target_from_pred':
                        write_jsonl([{'time': time_now, 'first_expr': str(args[0]), 'status': 'queue_empty'}], log_path, 'a')
                    else:
                        write_jsonl([{'time': time_now, 'first_expr': str(args[0]), 'second_expr': str(args[1]), 'status': 'queue_empty'}], log_path, 'a')
                logger.error(f'Consume time[queue_empty]: {time.time() - start} ...')
                if is_timeout:
                    raise TimeoutException("Operation timed out!")
                else:
                    raise Exception(f"Queue is empty! {time.time() - start}")
            
            # write_text(f'{func.__name__}[main3]: enter', save_path, 'a')
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                # print('666666666')
                # write_text(f'{func.__name__}[get]: enter1', save_path, 'a')
                result = []
                while True:
                    try:
                        serialized_data = result_queue.get_nowait()

                        # log_path2 = '/cpfs/align/chenliangbo/models/verl_checkpoints/ds7b_sys_dspposreuse_timeout/log.jsonl'
                        # output_path2 = '/cpfs/align/chenliangbo/models/verl_checkpoints/ds7b_sys_dspposreuse_timeout/output.bin'
                        # time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        # if func.__name__ == 'extract_target_from_pred':
                        #     write_jsonl([{'time': time_now, 'first_expr': str(args[0]), 'output_path': output_path2}], log_path2, 'w')
                        #     with open(output_path2, 'wb') as file:
                        #         file.write(serialized_data)
                        # else:
                        #     write_jsonl([{'time': time_now, 'first_expr': str(args[0]), 'second_expr': str(args[1]), 'output_path': output_path2}], log_path2, 'w')
                        #     with open(output_path2, 'wb') as file:
                        #         file.write(serialized_data)

                        # write_text(f'{func.__name__}[dill]: start. Got {len(serialized_data)} bytes. time: {time_now}', save_path, 'a')
                        result.append(load_expr(serialized_data))
                        # write_text(f'{func.__name__}[dill]: finish. Got {len(serialized_data)} bytes. time: {time_now}', save_path, 'a')
                    except:
                        break

                # write_text(f'{func.__name__}[get]: dill finish.', save_path, 'a')
                # write_text(f'{func.__name__}[get]: exit2', save_path, 'a')
                if func.__name__ == 'compare_single_extraction':
                    if len(result) == 0:
                        log_path = kwargs.get('log_path', None)
                        if log_path:
                            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            write_jsonl([{'time': time_now, 'first_expr': str(args[0]), 'second_expr': str(args[1]), 'status': 'length zero'}], log_path, 'a')
                    result = result[0] # IndexError: list index out of range
                if use_cache:
                    cache[cache_key] = dill.dumps(result)  # 序列化后存入缓存
                # time.sleep(0.00001)
                # logger.error(f'Consume time[Normal]: {time.time() - start} ...')
                return result
            except:
                # write_text(f'{func.__name__}[get]: enter2', save_path, 'a')
                log_path = kwargs.get('log_path', None)
                logger.error('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                if log_path:
                    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if func.__name__ == 'extract_target_from_pred':
                        write_jsonl([{'time': time_now, 'first_expr': str(args[0]), 'status': 'queue_get'}], log_path, 'a')
                    else:
                        write_jsonl([{'time': time_now, 'first_expr': str(args[0]), 'second_expr': str(args[1]), 'status': 'queue_get'}], log_path, 'a')
                import traceback
                traceback.print_exc()
                try:
                    logger.error(f'Consume time[queue_get]: {time.time() - start} ...')
                except:
                    import traceback
                    traceback.print_exc()
                # write_text(f'{func.__name__}[get]: exit3', save_path, 'a')
                raise Exception("Queue get error!")
        return wrapper
    return decorator


def init_shared_cache():
    """初始化共享缓存，必须在主进程中调用"""
    global _SHARED_CACHE 
    if _SHARED_CACHE is None:
        print('_SHARED_CACHE initialize ...')
        manager = Manager()
        _SHARED_CACHE = manager.dict()
    return _SHARED_CACHE

def shared_cache():
    init_shared_cache()
    """
    共享缓存装饰器
    :param get_key_func: 可选的函数，用于从参数生成缓存键
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if _SHARED_CACHE is None:
                raise RuntimeError("Shared cache not initialized. Call init_shared_cache() in main process first.")
            
            # 生成缓存键
            cache_key = get_cache_key(func.__name__, *args, **kwargs)

            # 检查缓存
            if cache_key in _SHARED_CACHE:
                return load_expr(_SHARED_CACHE[cache_key])
            
            try:
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                _SHARED_CACHE[cache_key] = dill.dumps(result)
                return result
            except Exception as e:
                raise e
        
        return wrapper
    return decorator


if __name__ == '__main__':
    @shared_cache()  # 使用自定义键生成
    def func2(a, b):
        print("func2 executing...")  # 这行只会在缓存未命中时执行
        return a + b

    @shared_cache()  # 使用自定义键生成
    def func3(a, b):
        print("func2 executing...")  # 这行只会在缓存未命中时执行
        return a - b

    def func1(x):
        result = func2(x, 10)  # 自动使用共享缓存
        return result * 2

    def worker(process_id):
        print(f"Process {process_id} - func1 result:", func1(process_id))
        # 第二次调用相同的参数会命中缓存
        print(f"Process {process_id} - cached func1 result:", func1(process_id))
    
    # 启动多个进程
    processes = []
    for i in range(3):
        p = multiprocessing.Process(target=worker, args=(i,))
        processes.append(p)
        p.start()
    
    # 等待所有进程完成
    for p in processes:
        p.join()
    
    # 打印最终缓存内容
    print("\nFinal shared cache:", dict(_SHARED_CACHE))
