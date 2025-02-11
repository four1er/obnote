---
{"title":"coroutine","auther":"four1er","created_at":"2025-02-06 00:21","last modify":"2025-02-06 00:21","file path":"C++/C++20/coroutine.md","tags":["coroutine","Cpp"],"dg-publish":true,"permalink":"/C++/C++20/coroutine/","dgPassFrontmatter":true,"created":"2025-02-10T01:26:10.742+08:00","updated":"2025-02-11T13:01:06.585+08:00"}
---


# Basic
## 协程的分类
根据**调度方式**的不同，协程可以分为**对称协程**和**非对称协程**，其主要区别在于：
- **对称协程**：任何一个协程都是相互独立且平等的，执行权可以在任意协程之间转移，比如 A 协程调用了 B 协程后，A 协程与 B 协程之后就没有任何关系了，B 协程让出执行权时，该执行权最终花落谁家都有可能；
- **非对称协程**：协程让出执行权的目标**只能是它的调用者**，即协程之间存在调用和被调用关系，比如 A 协程调用了 B 协程后，B 协程当需要让出执行权时一定是将执行权给了 A 协程。

根据**运行时栈**的不同协程可以分为**有栈协程**和**无栈协程**，其主要区别在于：
- **有栈协程**：协程上下文和变量等相关信息均保存在调用栈上；
- **无栈协程**：协程上下文和变量等相关信息保存在某个对象中，并不维护在调用栈上，比如维护在闭包中。

### 有栈协程（Todo）

### 无栈协程（Todo）

## 协程栈
协程栈分两种：共享栈与非共享栈。
### 非共享栈
在非共享栈模式下，每个非主协程有自己的栈。

> [!important]
> 该栈是在堆上分配的，而非系统栈。

但是主协程的栈仍然是系统栈，每两个协程的栈地址不相邻。
# 开源实现
## 1、Libco
Libco 是一个有栈非对称协程，那么首先要确定协程结构需要解决的问题：

## 2、TODO

# C++20 标准中的 Coroutine
## Data types
C++ 本身提供了一种数据类型：协程句柄 (coroutine handle)。它用于标识协程代码的特定实例，以及其所有内部变量及执行状态（例如：它当前是否正在执行，如果没有，则下次重新启动时它将从何处恢复）。它提供了一个 resume 方法，可以调用该方法来实际启动协程运行，或者在协程挂起后重新启动协程。
**协程的返回类型**。这是协程设置的用户将看到的唯一类型，在本文中，称之为面向用户的类型，它的一个实例是面向用户的对象。
协程的返回类型是协程的调用方实际与之交互的内容。在背后的实现中，它的主要数据类型称为**Promise**类型。它必须是一个 class，并且必须提供各种专门命名的方法来控制策略决策，比如是否启动或阻塞。
还有一种类型称之为 awaiters（awaitables）。这个类型的作用是为每个尝试暂停协程的单独事件设置策略。具体来说，每次协程执行 co_yield、co_await 或者 co_return 的时候，Promise type 将为该事件构造一个新的 awaiter，并且这个 awaiter 中的方法将会在某些情景发生时被调用（例如，协程是否会暂停自己，如果是它是将控制权转移给另一个协程还是返回给调用方；以及协程恢复的时候，返回给协程的值作为 co_yield 或者 co_await 的表达式）。标准库也提供了一些默认的 awaiter 类型：`std::suspend_always` 和 `std::suspend_never`。
## 从 hello world 开始
万事从 hello world 开始，以下实现了一个满足最低约束的例子：
```cpp
#include <coroutine>
#include <iostream>

class UserFacing {
 public:
  class promise_type {
   public:
    UserFacing get_return_object() { return UserFacing{}; }
    std::suspend_never initial_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception() {}
    std::suspend_always final_suspend() noexcept { return {}; }
  };
};

UserFacing demo_coroutine() {
  std::cout << "Hello World" << std::endl;
  co_return;
}

int main() {
  UserFacing demo_instance = demo_coroutine();
  return 0;
}
```
约束包括：
将 `UserFacing` 与 `promise` 类型关联起来了。
当编译器看到你定义了一个协程函数，并且它的返回类型是 `UserFacing`，那么编译器首先会去检查关联的 `promise` 类型是什么。
默认的实现方法是将 `UserFacing::promise_type` 设置成一个 typename。除了上述代码的实现方式外，还有一种实现方法是将这两者分离开，然后在 `UserFacing` 中声明 `promiss_type` 即可。类似：
```cpp
class UserFacing {
	public:
		using promise_type = TheActualPromiseType;
		//...
};
```
构造面向用户类型的实例以返回给用户。当协程实例化时，C++ 实现将负责创建承诺类型的实例并为其分配存储空间，但是最终由 `UserFacing` 来作为实际的接受对象。
这一步是由 `promise_type` 中的 `get_return_object` 方法来实现的。在上述这个例子中，`UserFacing` 是一个没有成员的对象，所以 `get_return_object` 以一种非常简单的方法构造了一个对象。
但是在更多的情况下，你可能会给这个面向用户的对象（`UserFacing` 对象）更多的成员信息，尤其是，这个面向用户的对象非常可能需要有访问 coroutine handle 和这个 promis 对象的权利，因为这两个对象在跟仍处于阻塞状态下的协程通信是不可或缺的。我们将在下面补全这个例子。
我们还需要考虑一个问题，当我们创建一个协程的时候，这个协程应当立马运行还是阻塞呢？这个是由 `initial_suspend()` 函数来决定的，这个函数必须返回一个 `awaiter` 类型。在上面这个例子中，我们使用了一个标准库提供的 `std::suspend_never` 来作为返回对象，这会导致协程不会在创建的时候阻塞，这也就意味着，这个协程在实际将控制权返回给 main 方法之前，已经实际运行起来了，并且打印出了 "hello world"。

> [!note]
> 这里我们如果将代码改成：
>```cpp
>class UserFacing {
> public:
>  class promise_type {
>   public:
>    UserFacing get_return_object() { return UserFacing{}; }
>    std::suspend_always initial_suspend() { return {}; }
>    void return_void() {}
>    void unhandled_exception() {}
>    std::suspend_always final_suspend() noexcept { return {}; }
>  };
>};
>```
> 那么将不会打印出 "hello world"，除非在 main 函数中使用 resume 去执行这个协程。

让我们来继续讨论协程返回的时候会发生什么。
这个例子中，协程并没有返回任何值（co_return 语句并没有任何参数）。所以我们在 `promise_type` 中提供了一个 `return_void()` 方法，这个方法将在 `co_return` 没有任何参数的时候执行。Ps：这个方法也会在代码脱离协程的时候执行。
当我们希望协程有一个非 void 的返回值的时候，我们需要去实现一个 `return_value()` 方法, 这个方法需要携带一个参数。（这个方法会在 `co_return` 带有一个值的时候被执行）

> 注意，我们必须实现这两个方法中的任一。如果我们都不实现或者都实现，将会报错！

接下来来讨论一个协程在执行过程中发生了异常的情况。如果 code 在执行过程中抛出了异常并且没有使用 try-catch 来捕获它，那么 `promise_type` 中的 `unhandled_exception()` 方法将会被调用，它会接受并保存这个异常，然后去执行一些代码去处理它。
在上面这个例子中，`unhandled_exception()` 并没有做任何事情，这意味着当有异常发生的时候，会被简单丢弃，然后这个协程进入类似被终止的状态。
最后来讨论一个协程被终止的时候会做什么。顾名思义，这个状态的处理是由 `final_suspend()` 函数来执行。这个函数非常类似 `initial_suspend()` 函数，除了 `final_suspend()` 函数还有一个 noexcept 修饰符。（原因在于这个阶段的异常会变得很难处理）
`final_suspend()` 将会在任何协程中止的情况下被调用，无论是正常运行完成还是因为异常而导致的中止状态。
上述这个例子中，`final_suspend()` 函数的返回类型是 `suspend_always`，这意味着当协程结束的时候，它将会进入阻塞状态，然后将控制权返回给调用者。

> `finial_suspend` 函数禁止返回值，这会导致本应终止的协程继续尝试去 running，这将会引发 crash。

## Co_await
在这一节中，我们将给出一个可以阻塞的协程。
C++ 提供了两个关键字以便于我们可以实现协程阻塞：1. `co_await` 2. `co_yield`. `co_await` 是更基础的一个关键字，`co_yield` 依赖 `co_await` 中的某些概念。
`co_await` 的核心思想是指明我们想要在某些情况下进行等待；而 `co_yield` 则是希望把某个值传递出去。
一种更通用的思路是：无论什么时候，当你给 `co_await` 传递一个操作数的时候，编译器都会将它转成一个 awaiter 对象，awaiter 对象将会告诉你如何去管理阻塞。
用一个例子来做简单的说明：
```cpp
UserFacing demo_coroutine() {
    std::cout << "we're about to suspend this coroutine" << std::endl;
    co_await std::suspend_always{};
    std::cout << "this won't be printed until after we resume" << std::endl;
    co_return;
}
```
这个协程中使用了 `co_await std::suspend_always{};`，这会导致这行后面的代码在没有被 resume 之前是肯定不会被执行到的。
对于更通用的场景，我们可能会希望我们的协程去阻塞等待某些事情，然后在事情发生的时候被 resumed。所以我们需要增加一些 handler code 来达到我们的目的。
有两个地方可以插入 handler code：
1. 如果 `promise_type` 中有一个名为 `await_transform()` 的方法，这个方法可以接受一个和传给 `co_await` 参数同样的类型。当这个函数被调用的时候，其参数会被替换成实际传给 `co_await` 的值。

	> 毫无疑问，这个函数可以被重载。

2. 你也可以选择重载 `co_await`。

> 如果同时有这两种请求，它将会安装顺序执行，也就是：`operator co_await(promise.await_transform(foo))`

让我们继续用一个例子进行说明：
```cpp
#include <chrono>
#include <coroutine>
#include <iostream>
#include <thread>

struct Event {};

class UserFacing {
 public:
  class promise_type {
   public:
    UserFacing get_return_object() { return UserFacing{}; }
    std::suspend_never initial_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception() {}
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_never await_transform(Event) {
      std::cout << "wait event happend..." << std::endl;
      // sleep 2s
      std::this_thread::sleep_for(std::chrono::seconds(2));
      return {};
    }
  };
};

UserFacing demo_coroutine() {
  std::cout << "we're about to suspend this coroutine" << std::endl;
  co_await Event{};
  std::cout << "this won't be printed until after we resume" << std::endl;
  co_return;
}

int main() {
  std::cout << "Main thread started." << std::endl;
  UserFacing demo_instance = demo_coroutine();
  std::cout << "Main thread resumed." << std::endl;
  return 0;
}

```

Output：
```shell
Main thread started.
we're about to suspend this coroutine
wait event happend…
this won't be printed until after we resume
Main thread resumed.
```

协程在等待两秒之后返回执行。
另外，我们还可以实现自定义的 awaiter，用来控制 `co_await` 的返回结果。比如：
```cpp
UserFacing demo_coroutine() {
    // You might set up co_await to return actual data
    ip_address addr = co_await async_dns_lookup("hostname.example.com");

    // Or a boolean indicating success or failure
    if (co_await attempt_some_network_operation(addr)) {
        std::cout << "success!" << std::endl;
    } else {
        std::cout << "failed, fall back to some other approach" << std::endl;
    }
}
```

## Resuming the coroutine
对一个协程的 resume 需要通过其 coroutine handle 操作来实现。
Coroutine handle 和 promise 对象同时创建，两者之间可以方便的互相转换。
- 从 promise 获取 handle，可以使用：`from_promise()
- 从 handle 获取 promise，可以使用：`promise()`

当我们构建一个新的协程对象的时候，系统调用 promise type 的 `get_return_object` 方法，在该方法中早已准备好了一个 promise 对象的引用：`*this`。所以我们可以使用 `*this` 来获取 coroutine handle。
在我们获取到了这个对象之后，我们可以把它传递给面向用户的对象的构造函数。代码示例如下：
```cpp
#include <chrono>
#include <coroutine>
#include <iostream>
#include <thread>

struct Event {};

class UserFacing {
 public:
  class promise_type {
   public:
    using handle_type = std::coroutine_handle<promise_type>;
    UserFacing get_return_object() {
      auto handle = handle_type::from_promise(*this);
      return UserFacing{handle};
    }
    std::suspend_never initial_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception() {}
    std::suspend_always final_suspend() noexcept { return {}; }
  };

  UserFacing(std::coroutine_handle<> handle) : m_coroutine_handle(handle) {}

  void resume() { m_coroutine_handle.resume(); }

 private:
  std::coroutine_handle<> m_coroutine_handle;
};

UserFacing demo_coroutine() {
  std::cout << "we're about to suspend this coroutine" << std::endl;
  co_await std::suspend_always{};
  std::cout << "this won't be printed until after we resume" << std::endl;
}

int main() {
  std::cout << "Main thread started." << std::endl;
  UserFacing demo_instance = demo_coroutine();
  std::cout << "Main thread resumed." << std::endl;
  std::cout << "Run demo_instance.resume()" << std::endl;
  demo_instance.resume();
  return 0;
}

```
Output:
```shell
Main thread started.
we're about to suspend this coroutine
Main thread resumed.
Run demo_instance.resume()
this won't be printed until after we resume
```

> [!note]
> 注意到，在上面的代码中，co_return 已经被删除了。我们之前之所以保留它，是因为我们需要把这个函数变成一个协程，所以通过 co_return 来进行显式声明。而现在我们已经有了 co_await, 所以 co_return 就不再需要了

## 管理协程状态
我们没有编写任何关于 promise 对象分配的代码，编译器已经替我们完成了这一切。所以需要担心它是否会被重复回收，在我们这样做之前，我们的协程系统可能会存在内置的内存泄漏。
所以我们需要对 coroutine handle 调用 `destroy()` 函数来解决这一问题。
我们可能很容易会想到，将这一函数放在面向用户对象的析构函数中，但是我们还需要考虑另一种情况：double-free。这个问题在用户对象被 copy 的时候很容易发生。
一个最简单的方法是我们将 copy constructor 和 copy assignment 设置为 delete。此外，还需要设置一个移动构造函数和一个移动赋值运算符，将移动对象中的协程句柄设置为 nullptr，这样当人们确实移动了面向用户的对象的时候，也不会发生 double-free 的问题。
```cpp
class UserFacing {
 public:
  class promise_type {
   public:
    using handle_type = std::coroutine_handle<promise_type>;
    UserFacing get_return_object() {
      auto handle = handle_type::from_promise(*this);
      return UserFacing{handle};
    }
    std::suspend_never initial_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception() {}
    std::suspend_always final_suspend() noexcept { return {}; }
  };

  void resume() { m_coroutine_handle.resume(); }

  UserFacing(std::coroutine_handle<> handle) : m_coroutine_handle(handle) {}

  UserFacing(UserFacing&& other) noexcept
      : m_coroutine_handle(other.m_coroutine_handle) {
    other.m_coroutine_handle = nullptr;
  }

  UserFacing& operator=(UserFacing&& other) noexcept {
    if (m_coroutine_handle) {
      m_coroutine_handle.destroy();
    }
    m_coroutine_handle = other.m_coroutine_handle;
    other.m_coroutine_handle = nullptr;
    return *this;
  }

  UserFacing(const UserFacing&) = delete;
  UserFacing& operator=(const UserFacing&) = delete;

  ~UserFacing() {
    if (m_coroutine_handle) {
      m_coroutine_handle.destroy();
    }
  }

 private:
  std::coroutine_handle<> m_coroutine_handle;
};
```
完整代码：
```cpp
#include <chrono>
#include <coroutine>
#include <iostream>
#include <thread>

struct Event {};

class UserFacing {
 public:
  class promise_type {
   public:
    using handle_type = std::coroutine_handle<promise_type>;
    UserFacing get_return_object() {
      auto handle = handle_type::from_promise(*this);
      return UserFacing{handle};
    }
    std::suspend_never initial_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception() {}
    std::suspend_always final_suspend() noexcept { return {}; }
  };

  void resume() { m_coroutine_handle.resume(); }

  UserFacing(std::coroutine_handle<> handle) : m_coroutine_handle(handle) {}

  UserFacing(UserFacing&& other) noexcept
      : m_coroutine_handle(other.m_coroutine_handle) {
    other.m_coroutine_handle = nullptr;
  }

  UserFacing& operator=(UserFacing&& other) noexcept {
    if (m_coroutine_handle) {
      m_coroutine_handle.destroy();
    }
    m_coroutine_handle = other.m_coroutine_handle;
    other.m_coroutine_handle = nullptr;
    return *this;
  }

  UserFacing(const UserFacing&) = delete;
  UserFacing& operator=(const UserFacing&) = delete;

  ~UserFacing() {
    if (m_coroutine_handle) {
      m_coroutine_handle.destroy();
    }
  }

 private:
  std::coroutine_handle<> m_coroutine_handle;
};

UserFacing demo_coroutine() {
  std::cout << "we're about to suspend this coroutine" << std::endl;
  co_await std::suspend_always{};
  std::cout << "this won't be printed until after we resume" << std::endl;
  co_return;
}

int main() {
  std::cout << "Main thread started." << std::endl;
  UserFacing demo_instance = demo_coroutine();
  std::cout << "Main thread resumed." << std::endl;
  std::cout << "Run demo_instance.resume()" << std::endl;
  //   demo_instance.resume();
  UserFacing demo_instance2{std::move(demo_instance)};
  demo_instance2.resume();
  return 0;
}
```
## 通过 co_yield 传递值
现在我们希望通过一段程序，返回一段 stream 给调用者。
类似：
```cpp
UserFacing demo_coroutine() {
	co_yield 100;
	for (int i = 1; i <= 3; i ++) {
		co_yield i;
	}
	co_yield 200;
}
```
然后给面向用户的对象一个 `next_value` 方法，它将返回 `100, 1, 2, 3, 200` 这样一段序列。
为了让 `co_yield` 能够合理的协程中运行起来，promise type 一定要提供一个方法：`yield_value`, 然后把我们需要 yield 出来的值作为参数。在上面这个例子中，我们需要定义这个函数为：`yield_value(int)`。
`yield_value` 的返回值就像 `co_await` 一样，它要不是一个 `awaiter` 类型，或者可以被传递进 `await_transform` 或 `operator co_await`。
在这里为了方便起见，我们继续使用 `std::suspend_always`。在这种情况下，每次执行到 co_yield 的时候，都会将控制权放回给 caller，所以我们可以在外部消费到传递出来的值。
简单来写：
```cpp
class UserFacing {
    // ...
    class promise_type {
        // ...
      public:
        int yielded_value;

        std::suspend_always yield_value(int value) {
            yielded_value = value;
            return {};
        }
    };
};
```
现在当运行到 `yield 100` 的时候，yielded_value 可以获取到 100，然后协程阻塞。
阻塞这个协程意味着如果我们想继续获取后续的值，我们需要调用 `handle.resume()`，然后使用 `handle.promise().yieled_value` 来获取到传递出来的值。
```cpp
class UserFacing {
    // ...
  public:
    int next_value() {
        handle.resume();
        return handle.promise().yielded_value;
    }
};
```
现在我们连续调用 5 次，将成功拿到 `100, 1, 2, 3, 200`
但是我们第六次调用这个函数的时候，会发生什么？
在第六次调用这个函数的时候，`co_yield 200;` 已经是最后一行语句了，接下来这个协程会终止，调用 `final_suspend()`，这会导致没有任何值会传递给 `yielded_value`，所以 `promise` 中依然会存有之前的 `yieled_value`: 200.
为了解决这个问题，我们需要在 resume 协程前将一些 dummy value 写入到 `yielded_value`, 然后当协程函数执行完的时候，它会将 dummy value 传递出去。
```cpp
#include <chrono>
#include <coroutine>
#include <iostream>
#include <optional>
#include <thread>

struct Event {};

class UserFacing {
 public:
  class promise_type;
  using handle_type = std::coroutine_handle<promise_type>;
  class promise_type {
   public:
    std::optional<int> yielded_value;
    UserFacing get_return_object() {
      auto handle = handle_type::from_promise(*this);
      return UserFacing{handle};
    }
    std::suspend_always initial_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception() {}
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always yield_value(int value) {
      yielded_value = value;
      return {};
    }
  };

  void resume() { m_coroutine_handle.resume(); }

  UserFacing(handle_type handle) : m_coroutine_handle(handle) {}

  UserFacing(UserFacing&& other) noexcept
      : m_coroutine_handle(other.m_coroutine_handle) {
    other.m_coroutine_handle = nullptr;
  }

  UserFacing& operator=(UserFacing&& other) noexcept {
    if (m_coroutine_handle) {
      m_coroutine_handle.destroy();
    }
    m_coroutine_handle = other.m_coroutine_handle;
    other.m_coroutine_handle = nullptr;
    return *this;
  }

  UserFacing(const UserFacing&) = delete;
  UserFacing& operator=(const UserFacing&) = delete;

  ~UserFacing() {
    if (m_coroutine_handle) {
      m_coroutine_handle.destroy();
    }
  }

  std::optional<int> next_value() {
    auto& promise = m_coroutine_handle.promise();
    promise.yielded_value = std::nullopt;
    m_coroutine_handle.resume();
    return promise.yielded_value;
  }

 private:
  handle_type m_coroutine_handle;
};

UserFacing demo_coroutine() {
  co_yield 100;
  for (int i = 1; i <= 3; i++) co_yield i;
  co_yield 200;
}

int main() {
  std::cout << "Main thread started." << std::endl;
  UserFacing demo_instance = demo_coroutine();
  while (std::optional<int> value = demo_instance.next_value()) {
    std::cout << "Got value: " << *value << std::endl;
  }
  return 0;
}
```
Output：
```shell
Main thread started.
Got value: 100
Got value: 1
Got value: 2
Got value: 3
Got value: 200
```
## 检查协程是否完成
在上面这段代码中有一个问题一直困惑着我，那就是如果当我们第七次调用 next_value 的时候，会发生什么呢？
此时协程已经执行完了，`final_suspend()` 也执行完了，此时继续 resume 这个协程是一个很明显的错误，这会导致一个 crash。
所以我们还需要做一点改进，在 `next_value()` 中首先需要去判断协程的运行状态。
```cpp
class UserFacing {
    // …
  public:
    std::optional<int> next_value() {
        auto &promise = handle.promise();
        promise.yielded_value = std::nullopt;
        if (!handle.done())
            handle.resume();
        return promise.yielded_value;
    }
};
```
## 通过 co_return 传递最终结果
可以通过 `return_value()` 来实现这一目的。
```cpp
class UserFacing {
    // …
  public:
    class promise_type {
      public:
        std::optional<std::string> returned_value;

        void return_value(std::string value) {
            returned_value = value;
        }
    };

    // …
    std::optional<std::string> final_result() {
        return handle.promise().returned_value;
    }
};
```
## 编写自定义的 `awaiters`
每次你的协程阻塞的时候都会去构造一个 `awaiter` 对象，并使用这个 `awaiter` 对象来控制这个阻塞的影响。
我们之前一直使用了标准库提供的 `std::suspend_always` 和 `std::suspend_never`, 现在是时候来实现我们自己的 `awaiter` 了。
Awaiter type 并不需要继承自任何类，它可以是任何类型，只需要实现三个方法即可：
```cpp
class Awaiter {
  public:
    bool await_ready();
    SuspendReturnType await_suspend(std::coroutine_handle<OurPromiseType> handle);
    ResumeReturnType await_resume();
};
```
首先来看第一个方法 `await_ready`, 它用于控制是否阻塞。如果 return true，那么协程将不会阻塞，继续执行。如果 return false，那么协程将会阻塞。

> 我们可以在 await_ready 方法中测试阻塞操作是否已经就绪，如果已经就绪，那就不需要阻塞了。

当 `await_ready` 返回 false 的时候，`await_suspend()` 函数被调用。该函数的参数是 handle，这意味着我们可以通过 `handle.promise()` 来获取 promise 对象。
`await_suspend()` 函数的返回值有几种选择：
- Void。此时协程将阻塞，并且控制权放回给上一个 resume 它的对象。
- Bool。
	- True。阻塞
	- False。不阻塞。
- Handle。协程阻塞，但是控制权不是立马回到上一个 resume 它的对象手中，而是会继续执行它返回对象的那个协程。

`await_resume` 在协程准备好运行的时候被调用。`await_resume` 的返回结果将会传递给协程本身，就像 `co_await` 或者 `co_yield` 表达式本身的值一样。
```cpp
UserFacing my_coroutine() {
    // ...

    std::optional<SomeData> result = co_await SomeNetworkTransaction();
    if (result) {
        // do something with the output
    } else {
        // error handling
    }

    // ...
}
```
就像使用同步编程一样。
Code：
```cpp
#include <coroutine>
#include <iostream>

class UserFacing {
 public:
  class promise_type;
  using handle_type = std::coroutine_handle<promise_type>;
  class promise_type {
   public:
    UserFacing get_return_object() {
      auto handle = handle_type::from_promise(*this);
      return UserFacing{handle};
    }
    std::suspend_always initial_suspend() { return {}; }
    void return_void() {}
    void unhandled_exception() {}
    std::suspend_always final_suspend() noexcept { return {}; }
  };

 private:
  handle_type handle;

  UserFacing(handle_type handle) : handle(handle) {}

  UserFacing(const UserFacing &) = delete;
  UserFacing &operator=(const UserFacing &) = delete;

 public:
  bool resume() {
    if (!handle.done()) handle.resume();
    return !handle.done();
  }

  UserFacing(UserFacing &&rhs) : handle(rhs.handle) { rhs.handle = nullptr; }
  UserFacing &operator=(UserFacing &&rhs) {
    if (handle) handle.destroy();
    handle = rhs.handle;
    rhs.handle = nullptr;
    return *this;
  }
  ~UserFacing() { handle.destroy(); }

  friend class SuspendOtherAwaiter;  // so it can get the handle
};

class TrivialAwaiter {
 public:
  bool await_ready() { return false; }
  void await_suspend(std::coroutine_handle<>) {}
  void await_resume() {}
};

class ReadyTrueAwaiter {
 public:
  bool await_ready() { return true; }
  void await_suspend(std::coroutine_handle<>) {}
  void await_resume() {}
};

class SuspendFalseAwaiter {
 public:
  bool await_ready() { return false; }
  bool await_suspend(std::coroutine_handle<>) { return false; }
  void await_resume() {}
};

class SuspendTrueAwaiter {
 public:
  bool await_ready() { return false; }
  bool await_suspend(std::coroutine_handle<>) { return true; }
  void await_resume() {}
};

class SuspendSelfAwaiter {
 public:
  bool await_ready() { return false; }
  std::coroutine_handle<> await_suspend(std::coroutine_handle<> h) { return h; }
  void await_resume() {}
};

class SuspendNoopAwaiter {
 public:
  bool await_ready() { return false; }
  std::coroutine_handle<> await_suspend(std::coroutine_handle<>) {
    return std::noop_coroutine();
  }
  void await_resume() {}
};

class SuspendOtherAwaiter {
  std::coroutine_handle<> handle;

 public:
  SuspendOtherAwaiter(UserFacing &uf) : handle(uf.handle) {}
  bool await_ready() { return false; }
  std::coroutine_handle<> await_suspend(std::coroutine_handle<>) {
    return handle;
  }
  void await_resume() {}
};

UserFacing demo_coroutine(UserFacing &aux_instance) {
  std::cout << "TrivialAwaiter:" << std::endl;
  co_await TrivialAwaiter{};
  std::cout << "ReadyTrueAwaiter:" << std::endl;
  co_await ReadyTrueAwaiter{};
  std::cout << "SuspendFalseAwaiter:" << std::endl;
  co_await SuspendFalseAwaiter{};
  std::cout << "SuspendTrueAwaiter:" << std::endl;
  co_await SuspendTrueAwaiter{};
  std::cout << "SuspendSelfAwaiter:" << std::endl;
  co_await SuspendSelfAwaiter{};
  std::cout << "SuspendNoopAwaiter:" << std::endl;
  co_await SuspendNoopAwaiter{};
  std::cout << "SuspendOtherAwaiter:" << std::endl;
  co_await SuspendOtherAwaiter{aux_instance};
  std::cout << "goodbye from coroutine" << std::endl;
}

UserFacing aux_coroutine() {
  while (true) {
    std::cout << "  aux_coroutine was resumed" << std::endl;
    co_await std::suspend_always{};
  }
}

int main() {
  UserFacing aux_instance = aux_coroutine();
  UserFacing demo_instance = demo_coroutine(aux_instance);
  while (demo_instance.resume())
    std::cout << "  suspended and came back to main()" << std::endl;
  std::cout << "and it's goodbye from main()" << std::endl;
}

```
Output:
```shell
TrivialAwaiter:
  suspended and came back to main()
ReadyTrueAwaiter:
SuspendFalseAwaiter:
SuspendTrueAwaiter:
  suspended and came back to main()
SuspendSelfAwaiter:
SuspendNoopAwaiter:
  suspended and came back to main()
SuspendOtherAwaiter:
  aux_coroutine was resumed
  suspended and came back to main()
goodbye from coroutine
and it's goodbye from main()
```
可以好好体会一下为什么是这个输出结果。
# Ref
1. https://en.cppreference.com/w/cpp/language/coroutines
2. https://www.chiark.greenend.org.uk/~sgtatham/quasiblog/coroutines-c++20/
3. https://km.woa.com/articles/show/580434?kmref=search&from_page=1&no=8
4. https://zplutor.github.io/2022/03/25/cpp-coroutine-beginner/
