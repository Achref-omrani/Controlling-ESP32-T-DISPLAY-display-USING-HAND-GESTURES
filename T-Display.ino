#include <TFT_eSPI.h> // Graphics and font library for ST7735 driver chip
#include <SPI.h>

TFT_eSPI tft = TFT_eSPI();  // Initialize TFT display
#define TFT_GREY 0x5AEB     // Define grey color for background
#define TFT_SELECTED 0x001F // Bright blue for the selected item
#define TFT_TEXT 0xFFFF     // Bright white for regular text

const char* menuItems[] = {"WiFi Scan", "Bluetooth Scan", "Saved Devices"};
const int menuSize = 3;
int currentSelection = 0; // Track the currently selected menu item

void setup() {
  Serial.begin(115200); // Initialize Serial communication
  tft.init();
  tft.setRotation(1);
  tft.fillScreen(TFT_GREY);
  tft.setTextSize(2); // Set text size for better visibility
  showMenu(); // Display the menu initially
}

void loop() {
  if (Serial.available() > 0) {
    String action = Serial.readStringUntil('\n');
    action.trim(); // Remove trailing newline or spaces
    processAction(action);
  }
}

// Function to display the menu
void showMenu() {
  tft.fillScreen(TFT_GREY); // Clear the screen
  for (int i = 0; i < menuSize; i++) {
    if (i == currentSelection) {
      tft.setTextColor(TFT_SELECTED, TFT_GREY); // Highlight selected item
    } else {
      tft.setTextColor(TFT_TEXT, TFT_GREY); // Regular menu item
    }
    tft.setCursor(10, 30 + (i * 40)); // Position menu items
    tft.println(menuItems[i]);
  }
}

// Function to scroll the menu smoothly
void scrollMenu(int direction) {
  int steps = 10; // Number of animation steps
  int stepDelay = 30; // Delay between steps in ms

  for (int step = 0; step <= steps; step++) {
    tft.fillScreen(TFT_GREY);
    for (int i = 0; i < menuSize; i++) {
      int offset = direction * (step * (40 / steps)); // Calculate smooth scroll offset
      if (i == currentSelection) {
        tft.setTextColor(TFT_SELECTED, TFT_GREY);
      } else {
        tft.setTextColor(TFT_TEXT, TFT_GREY);
      }
      tft.setCursor(10, 30 + (i * 40) - offset);
      tft.println(menuItems[i]);
    }
    delay(stepDelay); // Smooth animation
  }
  showMenu(); // Ensure menu is fully displayed after scrolling
}

// Process the received action
void processAction(String action) {
  if (action == "menu-item1") {
    currentSelection = 0; // Set selection to WiFi Scan
    scrollMenu(1); // Scroll down
    Serial.println("WiFi Scan selected");
  } else if (action == "menu-item2") {
    currentSelection = 1; // Set selection to Bluetooth Scan
    scrollMenu(1); // Scroll down
    Serial.println("Bluetooth Scan selected");
  } else if (action == "menu-item3") {
    currentSelection = 2; // Set selection to Saved Devices
    scrollMenu(1); // Scroll down
    Serial.println("Saved Devices selected");
  } else if (action == "menu-item4") {
    // Clear the menu and show a concise message
    tft.fillScreen(TFT_GREY);
    tft.setTextColor(TFT_SELECTED, TFT_GREY);
    tft.setCursor(10, 50);
    tft.println("Use gestures");
    tft.setCursor(10, 90);
    tft.println("to control menu");
    Serial.println("Entering gesture mode");
  } else {
    // Invalid action
    tft.fillScreen(TFT_GREY);
    tft.setTextColor(TFT_SELECTED, TFT_GREY);
    tft.setCursor(10, 60);
    tft.println("Invalid Action");
    Serial.println("Invalid action received");
  }
}
