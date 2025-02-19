netsh interface portproxy reset
netsh interface portproxy add v4tov4 listenaddress=192.168.1.8 listenport=19000 connectaddress=192.168.79.128 connectport=19000
netsh interface portproxy add v4tov4 listenaddress=192.168.1.8 listenport=19001 connectaddress=192.168.79.128 connectport=19001
netsh interface portproxy add v4tov4 listenaddress=192.168.1.8 listenport=19002 connectaddress=192.168.79.128 connectport=19002