
boolean [] RandomPosition () {
  boolean [] randomPos = new boolean[numOfItems];
  for (int i = 0; i < randomPos.length; i++) {
    randomPos[i] = random (0, 1) > 0.5;//random (0, possibleItemStates);
  }
  return randomPos;
}
float bestFitnessEverSeen = -1;
boolean [] bestPositionEver;
int [] previousFitnesses;
Fly [] flies;
int iterations = 0;

void UpdateFlies () {
  iterations++;
  Fly swarmBestFly = flies[0];
  int bestFitness = flies[0].GetFitness();
  int bestFlyIndex = 0;
  
  int [] fitnesses = new int[flies.length];
  for (int i = 1; i < flies.length; i++) {
    int fitness = flies[i].GetFitness ();
    if (bestFitness < fitness) {
      bestFitness = fitness;
      bestFlyIndex = i;
      swarmBestFly = flies[i];
    }
    fitnesses[i] = flies[i].GetFitness(); 
  }
  boolean [] bestFlyPos = swarmBestFly.position;
  
  boolean [][] newPositions = new boolean [flies.length][bestFlyPos.length];
  
  if (bestFitness > bestFitnessEverSeen) {
    bestFitnessEverSeen = bestFitness;
    bestPositionEver = bestFlyPos;
  }
  
  for (int i = 0; i < flies.length; i++) {
    Fly thisFly = flies[i];
    
    int bestNFitness = -1;
    int bestNIndex = -1;
    if (bestFlyIndex == i) {
      continue;
    }
    
    for (int j = i - 3; j < i + 4; j++) {
      //if (j == i) {
      //  continue;
      //}
      int wrapedIndex = (j + flies.length) % flies.length;
      if (fitnesses[wrapedIndex] > bestNFitness) {
        bestNFitness = fitnesses[wrapedIndex];
        bestNIndex = wrapedIndex;
      }
    }
    
    boolean [] bestNPos = flies[bestNIndex].position;
    if (bestNIndex == i) {
        flies[i].chanceToRelocate = 0;//(1-(fitnesses[i] / bestFitnessEverSeen)) * 0.03;
    }
    if (random(0, 1) > flies[i].chanceToRelocate) {//flies[i].chanceToRelocate) {//flies[i].chanceToRelocate) {
      for (int j = 0; j < numOfItems; j++) {
       newPositions[i][j] = random (0, 1) > 0.5 ? thisFly.position[j] : bestNPos[j]; //: (random (0, 1) > 0.5 ? bestFlyPos[j] : bestNPos[j]);
      }
      //if (fitnesses[i] < previousFitnesses[i]) {
        flies[i].chanceToRelocate = lerp (flies[i].chanceToRelocate, 0.4, 0.015);
      //}
    } else {
      newPositions[i] = RandomPosition ();
      flies[i].chanceToRelocate = 0;
    }
  }
  
  for (int i = 0; i < flies.length; i++) {
   flies[i].position = newPositions[i]; 
  }
  previousFitnesses = fitnesses;
}
class Fly {
  boolean [] position;
  float chanceToRelocate = 0.0001;
  
  
  
  Fly () {
    position = RandomPosition ();
  }
  int GetFitness () {
    int profitSoFar = 0;
    int [] weightsSoFar = new int [numOfSacks];
    for (int i = 0; i < numOfItems; i++) {
      //int itemSack = (int)position[i];
      //if (itemSack >= numOfSacks || itemSack < 0) {//Out of bounds means its not included in the sack
      //  continue;
      //}
      boolean state = position[i];
      if (!state) {//Out of bounds means its not included in the sack
        continue;
      }
      profitSoFar += itemProfits[i];
      for (int itemSack = 0; itemSack < numOfSacks; itemSack++) {
        weightsSoFar[itemSack] += itemWeights[itemSack][i];
      }
    }
    int weightOverCapacity = 0;
    for (int i = 0; i < numOfSacks; i++) {
      float overCapacity = sackCapacities[i] - weightsSoFar[i];
      if (overCapacity < 0) {
        weightOverCapacity -= overCapacity;
        //return 0;
        profitSoFar *= 0.8;
      }
      //if (sackCapacities[i] < weightsSoFar[i]) {
      //  return 0;
      //}
    }
    //if (weightOverCapacity != 0) {
    //  profitSoFar *= 0.002;
    //}
    //int prof = max((int)(profitSoFar - weightOverCapacity*2), 0);
    //if (prof >= 5557) {
    //  println ("profitSoFar: " + profitSoFar);
    //}
    return profitSoFar;
  }
  void PrintPosition () {
    for (int i = 0; i < position.length; i++) {
      print ((boolean)position[i] + ", ");
    }
    println();
  }
}