int numOfSacks;
int numOfItems;
int [] sackCapacities;
int [][] itemWeights;
int [] itemProfits;
int possibleItemStates; //Number of bags + 1 (last position means it gets left out)

int bestPossible;

void ImportFile (String fileName) {
  String [] strings = loadStrings (fileName);
  String [] str = join (strings, " ").split(" ");
  int [] nums = new int [str.length];
  int nextNumIndex = 0;
  
  
  for (int i = 0; i < str.length; i++) {
   if (str[i] == " " || str[i].length() == 0) {
     continue;
   }
   nums[nextNumIndex] = int(str[i]);
   nextNumIndex++;
  }
  
  int intReadingIndex = 0;
  numOfSacks = nums[intReadingIndex];
  intReadingIndex++;
  numOfItems = nums[intReadingIndex];
  intReadingIndex++;
  possibleItemStates = 2;//numOfSacks + 1;
  
  sackCapacities = new int[numOfSacks];
  itemWeights = new int[numOfSacks][numOfItems];
  itemProfits  = new int[numOfItems];
  
  for (int i = 0; i < numOfItems; i++) {
    itemProfits[i]= nums[intReadingIndex];
    intReadingIndex++;
  }
  for (int i = 0; i < numOfSacks; i++) {
    sackCapacities[i] = nums[intReadingIndex];
    intReadingIndex++;
  }
  for (int i = 0; i < numOfSacks; i++) {
    for (int j = 0; j < numOfItems; j++) {
      itemWeights[i][j] = nums[intReadingIndex];
      intReadingIndex++;
    }
  }
  //printArray(sackCapacities);
  //printArray(itemWeights[itemWeights.length - 1]);
  bestPossible = nums[intReadingIndex];
}


//int TotalWeightOf (int dna) {
//  int totalWeight = 0;
//  for (int i = 0; i < napSackProfit.length; i++) {
//    if (((dna >> i) & 1) == 1) {
//      totalWeight += napSackWeights[i];
//    }
//  }
//  return totalWeight;
//}
//int TotalProfitOf (int dna) {
//  if (dna < 0)
//    return 0;
//  int totalProfit = 0;
//  for (int i = 0; i < napSackProfit.length; i++) {
//    if (((dna >> i) & 1) == 1) {
//      totalProfit += napSackProfit[i];
//    }
//  }
//  return totalProfit;
//}
//int searchSpace = 1 << napSackProfit.length;