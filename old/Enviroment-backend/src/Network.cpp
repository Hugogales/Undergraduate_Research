#include "Network.h"
#include <iostream>

Network::Network(int port, StateCallback state_callback, InputCallback input_callback)
    : acceptor_(io_context, tcp::endpoint(tcp::v4(), port)),
      state_callback_(state_callback),
      input_callback_(input_callback),
      running(false) {}

Network::~Network() {
    stop();
}

void Network::start() {
    running = true;
    do_accept();
    io_thread = std::thread([this]() { io_context.run(); });
    std::cout << "Network server started on port " << acceptor_.local_endpoint().port() << std::endl;
}

void Network::stop() {
    if (running) {
        running = false;
        io_context.stop();
        if (io_thread.joinable()) {
            io_thread.join();
        }
        std::cout << "Network server stopped." << std::endl;
    }
}

void Network::do_accept() {
    acceptor_.async_accept(
        [this](boost::system::error_code ec, tcp::socket socket) {
            if (!ec && running) {
                std::cout << "Client connected: " << socket.remote_endpoint() << std::endl;
                std::thread(&Network::handle_session, this, std::move(socket)).detach();
            }
            if (running) {
                do_accept();
            }
        }
    );
}

void Network::handle_session(tcp::socket socket) {
    try {
        while (running) {
            // Send game state
            std::string game_state = state_callback_();
            boost::asio::write(socket, boost::asio::buffer(game_state + "\n"));

            // Read input
            boost::asio::streambuf buf;
            boost::system::error_code error;
            boost::asio::read_until(socket, buf, "\n", error);
            if (error) {
                std::cerr << "Read error: " << error.message() << std::endl;
                break;
            }
            std::istream is(&buf);
            std::string input;
            std::getline(is, input);

            if (!input.empty()) {
                input_callback_(input);
            }

            // Control the update rate (~60 FPS)
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
    catch (std::exception& e) {
        std::cerr << "Session error: " << e.what() << std::endl;
    }
}
