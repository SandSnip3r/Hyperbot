# Private server

## IP Addresses

### Public
The private server depends on my public IP address. That address is currently 73.63.155.251.
If this address changes, change:
- SRO client's division info
  - C:\Users\Victor\Documents\Development\PK2 Tools\DivisionInfo.exe
    - No need to choose a specific file, just the directory containing sro_client.exe
  - Now that Hyperbot is also over in linux, make sure to copy the updated Media.pk2
    - From `~/sro_files/sro_client$` in Ubuntu 22
    - `cp /mnt/c/Users/Victor/Documents/Development/Daxter\ Silkroad\ server\ files/Silkroad\ Client/Media.pk2`
- Hard coded address in module proxy in VM
  - Use Visual Studio
  - C:\Users\Victor\Desktop\edxModuleProxy2\src\Backend\Proxy\Client\ClientToGateway.cs
    - Line 120
    - Rebuild "Backend"
    - Right now, I'm using .net 5.0
    - Restart server
    - Backend shortcut in Server folder already points to newly build binary

### DESKTOP-JJC10JD
The server also depends on my desktop's current address as assigned by the router. That address is currently 192.168.1.8.
If this address changes, change:
- Netgear port forwarding
- C:\Users\Victor\Documents\Development\misc\netsh.bat
  - Run as admin
