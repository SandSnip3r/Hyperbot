/*class BotConnection {
private:

	struct BotData {
		std::vector<uint8_t> data;
		StreamUtility pending_stream;

		BotData() {
			data.resize(Config::DataMaxSize + 1);
		}
	};

	//Loader that holds sro process
	Loader *loader;

	//Accepts TCP connections
	boost::asio::ip::tcp::acceptor acceptor;

	//Connections
	std::map<boost::shared_ptr<boost::asio::ip::tcp::socket>, boost::shared_ptr<BotData> > sockets;

	//Starts accepting new connections
	void PostAccept(uint32_t count = 1) {
		for(uint32_t x = 0; x < count; ++x) {
			//The newly created socket will be used when something connects
			boost::shared_ptr<boost::asio::ip::tcp::socket> s(boost::make_shared<boost::asio::ip::tcp::socket>(io_service));
			acceptor.async_accept(*s, boost::bind(&BotConnection::HandleAccept, this, s, boost::asio::placeholders::error));
		}
		std::cout << "PostAccept, Loadering\n";
		std::string kSilkroadPath = "C:\\Program Files (x86)\\AtomixOnline\\";
		loader = new Loader(kSilkroadPath);
		try {
			loader->launch();
		}
		catch (std::exception &ex) {
			std::cout << "Loader::launch failed: " << ex.what() << '\n';
			delete loader;
		}
	}

	//Handles new connections
	void HandleAccept(boost::shared_ptr<boost::asio::ip::tcp::socket> s, const boost::system::error_code & error) {
		//Error check
		if(!error) {
			std::cout << "Bot/Analyzer connected" << std::endl;

			//Disable nagle
			s->set_option(boost::asio::ip::tcp::no_delay(true));

			//Add the connection to the list
			boost::shared_ptr<BotData> temp = boost::make_shared<BotData>();
			sockets[s] = temp;

			s->async_read_some(boost::asio::buffer(&temp->data[0], Config::DataMaxSize), boost::bind(&BotConnection::HandleRead, this, s, boost::asio::placeholders::bytes_transferred, boost::asio::placeholders::error));

			//Post another accept
			PostAccept();
		}
	}

	//Handles incoming packets
	void HandleRead(boost::shared_ptr<boost::asio::ip::tcp::socket> s, size_t bytes_transferred, const boost::system::error_code & error) {
		std::map<boost::shared_ptr<boost::asio::ip::tcp::socket>, boost::shared_ptr<BotData> >::iterator itr = sockets.find(s);
		if(itr != sockets.end()) {
			if(error) {
				if(s) {
					//Shutdown and close the connection
					boost::system::error_code ec;
					s->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
					s->close(ec);
				}

				//Remove the socket from the list
				sockets.erase(s);
			} else {
				//Extract structure objects
				std::vector<uint8_t> & data = itr->second->data;
				StreamUtility & pending_stream = itr->second->pending_stream;

				//Write the received data to the end of the stream
				pending_stream.Write<uint8_t>(&data[0], bytes_transferred);

				//Total size of stream
				int32_t total_bytes = pending_stream.GetStreamSize();

				//Make sure there are enough bytes for the packet size to be read
				while(total_bytes > 2) {
					//Peek the packet size
					uint16_t required_size = pending_stream.Read<uint16_t>(true) + 6;

					//See if there are enough bytes for this packet
					if(required_size <= total_bytes) {
						StreamUtility r(pending_stream);

						//Remove this packet from the stream
						pending_stream.Delete(0, required_size);
						pending_stream.SeekRead(0, Seek_Set);
						total_bytes -= required_size;

						uint16_t size = r.Read<uint16_t>();
						uint16_t opcode = r.Read<uint16_t>();
						uint8_t direction = r.Read<uint8_t>();
						r.Read<uint8_t>();

						//Remove the header
						r.Delete(0, 6);
						r.SeekRead(0, Seek_Set);

						if(opcode == 1 || opcode == 2) {
							uint16_t real_opcode = r.Read<uint16_t>();

							//Block opcode
							if(opcode == 1) {
								BlockedOpcodes[real_opcode] = true;
								std::cout << "Opcode [0x" << std::hex << std::setfill('0') << std::setw(4) << real_opcode << "] has been blocked" << std::endl << std::dec;
							}
							//Remove blocked opcode
							else if(opcode == 2) {
								boost::unordered_map<uint16_t, bool>::iterator itr = BlockedOpcodes.find(real_opcode);
								if(itr != BlockedOpcodes.end()) {
									BlockedOpcodes.erase(itr);
									std::cout << "Opcode [0x" << std::hex << std::setfill('0') << std::setw(4) << real_opcode << "] has been unblocked" << std::endl << std::dec;
								}
							}
						} else {
							//Silkroad
							if(direction == 2 || direction == 4) {
								InjectSilkroad(opcode, r, direction == 4 ? true : false);
							}
							//Joymax
							else if(direction == 1 || direction == 3) {
								InjectJoymax(opcode, r, direction == 3 ? true : false);
							}
						}
					} else {
						//Not enough bytes received for this packet
						break;
					}
				}

				//Read more data
				s->async_read_some(boost::asio::buffer(&data[0], Config::DataMaxSize), boost::bind(&BotConnection::HandleRead, this, s, boost::asio::placeholders::bytes_transferred, boost::asio::placeholders::error));
			}
		}
	}

public:

	//Constructor
	BotConnection(uint16_t port) : acceptor(io_service, boost::asio::ip::tcp::endpoint(boost::asio::ip::tcp::v4(), port)) {
		PostAccept();
	}

	//Destructor
	~BotConnection() {
	}

	//Sends packets to all connections
	void Send(PacketContainer & container, uint8_t direction) {
		StreamUtility & r = container.data;

		StreamUtility w;
		w.Write<uint16_t>(r.GetReadStreamSize());
		w.Write<uint16_t>(container.opcode);
		w.Write<uint8_t>(direction);
		w.Write<uint8_t>(container.encrypted);
		
		while(r.GetReadStreamSize())
			w.Write<uint8_t>(r.Read<uint8_t>());

		//Reset the read index
		r.SeekRead(0, Seek_Set);

		//Iterate all connections
		std::map<boost::shared_ptr<boost::asio::ip::tcp::socket>, boost::shared_ptr<BotData> >::iterator itr = sockets.begin();
		while(itr != sockets.end()) {
			//Send the packet
			boost::system::error_code ec;
			boost::asio::write(*itr->first, boost::asio::buffer(w.GetStreamPtr(), w.GetStreamSize()), boost::asio::transfer_all(), ec);
			
			//Next
			++itr;
		}
	}

	void Stop() {
		boost::system::error_code ec;

		//Iterate all connections
		std::map<boost::shared_ptr<boost::asio::ip::tcp::socket>, boost::shared_ptr<BotData> >::iterator itr = sockets.begin();
		while(itr != sockets.end()) {
			//Shutdown and close the connection
			itr->first->shutdown(boost::asio::ip::tcp::socket::shutdown_both, ec);
			itr->first->close(ec);

			//Next
			++itr;
		}

		sockets.clear();
	}
};*/