# with heltec board from
# https://github.com/Heltec-Aaron-Lee/WiFi_Kit_series/releases/download/0.0.4/package_heltec_esp8266_index.json
# and library https://github.com/HelTecAutomation/Heltec_ESP32

# device: wifi kit 8 (ESP8266 diy more)


#include <Wire.h>  // Only needed for Arduino 1.6.5 and earlier
#include "heltec.h" // alias for `#include "SSD1306Wire.h"`



void setup() {
  Serial.begin(115200);
  
  Heltec.begin(true /*DisplayEnable Enable*/, true /*Serial Enable*/);
  Heltec.display->init();
  Heltec.display->flipScreenVertically();
  Heltec.display->setFont(ArialMT_Plain_10);


}

void loop() { 

  String tmp = Serial.readStringUntil('\n');
 // Serial.println("got something");
  
  if (tmp.substring(0,9)==String("--print--")) {
    Serial.println("got print");
    Heltec.display->clear();
    Heltec.display->drawString(0,0,getValue(tmp,'|', 1));
    Heltec.display->drawString(0,10,getValue(tmp,'|', 2));
    Heltec.display->drawString(0,20,getValue(tmp,'|', 3));
    Heltec.display->display();
    
  }
  
}

String getValue(String data, char separator, int index)
{
    int found = 0;
    int strIndex[] = { 0, -1 };
    int maxIndex = data.length() - 1;

    for (int i = 0; i <= maxIndex && found <= index; i++) {
        if (data.charAt(i) == separator || i == maxIndex) {
            found++;
            strIndex[0] = strIndex[1] + 1;
            strIndex[1] = (i == maxIndex) ? i+1 : i;
        }
    }
    return found > index ? data.substring(strIndex[0], strIndex[1]) : "";
}
