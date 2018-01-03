float millsAtStart;

void setup () {
  ImportFile ("KnapSackInfo.txt");
  millsAtStart = millis();
  
  flies = new Fly [2000];
  for (int i = 0; i < flies.length; i++) {
   flies[i] = new Fly (); 
  }
  previousFitnesses = new int [flies.length];
  
  frameRate(99999);
}
void draw () {
  
  for (int i = 0; i < 1;i++) {
    UpdateFlies ();
    if (bestFitnessEverSeen >= bestPossible) {
      println ("Found result in: " + (millis () - millsAtStart) + " Millis");
      println ("Fitness: " + bestFitnessEverSeen);
      println ("Iterations: " + iterations);
      println ("PerfectState: ");
      println ("FEs: " + iterations * flies.length);
      
      Fly flyForPrinting = new Fly ();
      flyForPrinting.position = bestPositionEver;
      flyForPrinting.PrintPosition();
      
      exit ();
      return;
    }
  }
  println ("Iterations: " + iterations + ", BestEver: " + bestFitnessEverSeen);
}