# aiku
AI system to recognize images and write haikus inspired on them

Project presented at Uniarts' Research Pavilion #4. Helsinki. 2021

The Raspi downloads random images from wikipedia and uses the google coral TPU to run inferences on then. It detects objects, which serve as prompt to inspire the generation of Haiku (done remotely on PC)

## Components

* Raspi 3 with Google Coral TPU and 3.2 inch display
* Wifi kit 4 (ESP8266 board with oled display)
* PC computer with Nvidia RTX2060 running gpt-2 trained on haikus.

### network

IPs are hardwired to my zerotier setup
