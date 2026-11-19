// Turntable + X-ray trigger for the Samsung S-series X-ray panel controller.
// https://github.com/gamerpaddy/Samsung-S-Xraypanel-Controller
//
// Released under The Unlicense (public domain).
// Needs the AccelStepper library (Mike McCauley, GPL-3.0 or commercial license), which is NOT bundled:
// install it from the Arduino Library Manager.
//
// Serial, 9600 baud, one command per line:
//   CW<deg>    rotate clockwise by <deg> degrees, answers "OK" when finished
//   CCW<deg>   rotate counter-clockwise, answers "OK" when finished
//   C<ms>      pin 12 HIGH for <ms> milliseconds (X-ray trigger), then LOW
//
// Wiring: pin 12 = trigger output (drive your X-ray source ONLY through a properly isolated relay or
// optocoupler stage with your own safety interlocks), pin 8 = STEP, pin 9 = DIR of a step/dir motor driver.
// X-rays are dangerous. Use at your own risk.

#include <AccelStepper.h>

const int outputPin = 12;
const int stepPin = 8;
const int dirPin = 9;

// Define stepper motor
AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);

// Variables for stepper configuration
float stepsPerDegree = 9.736; // Adjust this value based on your motor/gearing
float maxSpeed = 3200.0;      // Adjust this value (steps per second)
float acceleration = maxSpeed*2;   // Adjust this value (steps per second^2)

String input = "";

void setup() {
  pinMode(outputPin, OUTPUT);
  digitalWrite(outputPin, LOW);

  stepper.setMaxSpeed(maxSpeed);
  stepper.setAcceleration(acceleration);

  Serial.begin(9600);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (input.length() > 0) {
        if (input.startsWith("CW")) {
          float angle = input.substring(2).toFloat();
          long steps = round(angle * stepsPerDegree);
          stepper.move(steps);
          while (stepper.distanceToGo() != 0) {
            stepper.run();
          }
          Serial.println("OK");
        }
        else if (input.startsWith("CCW")) {
          float angle = input.substring(3).toFloat();
          long steps = -round(angle * stepsPerDegree);
          stepper.move(steps);
          while (stepper.distanceToGo() != 0) {
            stepper.run();
          }
          Serial.println("OK");
        }
        else if (input.startsWith("C")) {
          int duration = input.substring(1).toInt();
          if (duration > 0) {
            digitalWrite(outputPin, HIGH);
            delay(duration);
            digitalWrite(outputPin, LOW);
          }
        }
        input = "";
      }
    } else {
      input += c;
    }
  }
}
