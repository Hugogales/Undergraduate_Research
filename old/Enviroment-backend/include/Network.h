#pragma once
#include <boost/asio.hpp>
#include <thread>
#include <functional>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>

using boost::asio::ip::tcp;

// Callback types
using StateCallback = std::function<std::string()>;
using InputCallback = std::function<void(const std::string&)>;

class Network {
public:
    Network(int port, StateCallback state_callback, InputCallback input_callback);
    ~Network();

    void start();
    void stop();

private:
    void do_accept();
    void handle_session(tcp::socket socket);

    boost::asio::io_context io_context;
    tcp::acceptor acceptor_;
    std::thread io_thread;

    StateCallback state_callback_;
    InputCallback input_callback_;

    bool running;
    std::mutex mtx;
};
