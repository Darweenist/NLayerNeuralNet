import java.io.*;
import java.util.Arrays;

/*
 * @author Dawson Chen
 * @since 2.18.2020
 * The Network class implements a neural network with 1 hidden layer, which allows the user to determine the number of activations
 * in the input, hidden, and output layers. It performs gradient descent given hyperparameters that are specified by the user,
 * such as the rate at which it trains and when to stop training.
 * This assignment is done for Dr. Nelson's AT Neural Network class (P2).
 */
public class NetworkDebug
{
   /*
    * This is a 2d array of doubles that stores all of the network's activations for the current training set.
    * The first dimension represents layers, and the second dimension represents the activations in a layer.
    */
   private double[][] activations;

   /*
    * This is a 3d array of doubles that stores all of the network's weights across all training sets.
    * The first dimension represents layers. The second dimension represents the activations in the input, or left-hand side layer.
    * The third dimension represents the output, or right-hand side layer. The final output layer is not included,
    * as it is simply one value.
    */
   private double[][][] weights;

   private double[] Fi; // Stores the final output values for the current training set
   private double[] Ti; // Stores the truth values of the current training set
   private double[] Ei; // Stores the errors of the final output values for the current training set with respect to the truth values

   private double weightsUpperBound; // The upper bound for random initialization of the weights
   private double weightsLowerBound; // The lower bound for random initialization of the weights

   private double learningFactor; // Determines how fast the network learns, or how fast the weights are to change
   private int maxIterations;     // The maximum number of times the network trains before giving up
   private double errorThreshold; // If the output has an error below this threshold, the network is said to be convergent

   private static BufferedReader configReader; // A reader that takes in input from the user about the configuration of the network
   private static BufferedReader inputsReader; // A reader that takes in input from the user about the input layer values

   private int numInputs;  // Stores the number of input activations in the network
   private int numHidden;  // Stores the number of hidden activations in the network's single hidden layer
   private int numOutputs; // Stores the number of output activations in the network
   private int numLayers;  // Stores the number of layers in the network

   private int[] activationDimensions; // Stores the dimensions of the layers of activations, summarizing the above 4 variables

   private int maxAct; // Stores the maximum number of activations in a layer of the network

   private int maxLeftActivations;  // Maximum number of activations in any input, or left-hand side layer of activations
   private int maxRightActivations; // Maximum number of activations in any output, or right-hand side layer of activations

   /*
    * The following instance variables are all to be used in training, in the procedure of executing gradient descent.
    * The specific mathematical functions and relationships of these variables is described clearly in the design document,
    * titled "Minimization of the Error Function for Two Layers".
    */
   private double[] hj;
   private double[] omegai;
   private double[] thetai;
   private double[] psii;
   private double[] thetaj;
   private double[] omegaj;
   private double[] uppercasePsij;
   private double[][] derivativeji;
   private double[][] derivativekj;
   private double[][][] deltaWeight;

   /**
    * This method is run at runtime and initializes a Network object to run the network. First, this method
    * asks for input from the user regarding what file the network configurations, weights, and inputs are written.
    * These inputted file path names must be given with the current directory being src.
    * Next, this method initializes BufferedReader's for each of these files and calls the setup functions to
    * initialize the weights and activations array. Then, this method updates all the activations based on the weights,
    * input activations and the threshold function.
    *
    * @param args is the array of strings which stores file paths passed by command line when starting a program.
    *             The order of the input must be configuration file name, weights file name, inputs file name
    * @throws IOException if the file name inputted does not represent a file that exists
    */
   public static void main(String[] args) throws IOException
   {
      NetworkDebug network = new NetworkDebug(); // Initializes a Network object, on which following methods are run
      /*
       * The following lines initializes 3 BufferedReader's to read in the user inputted structure of the network,
       * the input (0th) layer's activation values, and all of the weights.
       */
      String configFile;      // Path to file containing configuration of the network
      String weightsFile;     // Path to file containing initial weights of the network
      String inputsFile;      // Path to file containing input (0th) layer activations

      if (args.length == 1)
      {
         /*
          * If only one argument given, initialize configFile name as the argument, rest are default
          */
         configFile = args[0];
         weightsFile = "weights";
         inputsFile = "all";
      }
      else if (args.length == 2)
      {
         /*
          * If two arguments given, initialize configFile as first argument, weightsFile as second, third is default
          */
         configFile = args[0];
         weightsFile = args[1];
         inputsFile = "all";
      }
      else if (args.length == 3)
      {
         /*
          * If three arguments given, initialize configFile name as the argument, weightsFile as second, inputsFile as third
          */
         configFile = args[0];
         weightsFile = args[1];
         inputsFile = args[2];
      }
      else
      {
         /*
          * If no arguments given, initialize config, weights, and inputs file as default
          */
         configFile = "config";
         weightsFile = "weights";
         inputsFile = "all";
      }

      /*
       * Initialize BufferedReader's to read input from config and inputs files
       */
      configReader = new BufferedReader(new FileReader(new File(configFile)));

      inputsReader = new BufferedReader(new FileReader(new File(inputsFile)));

      String inputsLine1 = inputsReader.readLine();
      int numTrainingSets = Integer.parseInt(inputsLine1.substring(inputsLine1.indexOf("=") + 1));

      /*
       * The length of the first dimension of the activations array is the number of layers,
       * and the length the second dimension is the maximum number of nodes in a layer. The maximum number of nodes
       * are used rather than the exact number of nodes because it is impossible to initialize an array in java
       * with different lengths in the second dimension.
       */
      String configLine1 = configReader.readLine();                                       // Reads the first line of the config file
      network.numLayers = Integer.parseInt(configLine1.substring(configLine1.indexOf("=") + 1));  // Finds number of layers of activations
      /*
       * Note: the number of layers inputted by the user refers to the number of layers of weights, therefore it does not account for
       * the output layer.
       */
      network.activationDimensions = new int[network.numLayers]; // Represents the lengths of each layer

      String line2 = configReader.readLine();    // Reads the second line of the configuration file
      int index = line2.indexOf("=") + 1;        // Numbers after this "=" are layer lengths

      line2 += " "; // Adjusts the line to make layer lengths more easily decipherable when using indexOf to find the next space
      /*
       * This loop fills activationDimensions with inputted lengths of each layer from the configuration file.
       * The length of each layer is separated by white spaces.
       */
      for (int layer = 0; layer < network.numLayers; layer++)
      {
         network.activationDimensions[layer] = Integer.parseInt(line2.substring(index, line2.indexOf(" ", index)));
         index = line2.indexOf(" ", index + 1) + 1;
      }

      network.numInputs = network.activationDimensions[0]; // Sets the number of inputs to the first layer's activation dimensions

      network.numHidden = network.activationDimensions[1]; // Sets the number of hidden activations to be the second layer's activation dimensions

      network.numOutputs = network.activationDimensions[network.numLayers - 1]; // Sets the number of outputs to the last layer's activation dimensions

      network.Fi = new double[network.numOutputs];

      network.Ti = new double[network.numOutputs];

      network.Ei = new double[network.numOutputs];

      network.maxAct = 0; // Represents the maximum number of activations in a layer in activations, finds it by looping through all
      for (int numActs : network.activationDimensions)
      {
         if (numActs > network.maxAct)
            network.maxAct = numActs;
      }
      network.activations = new double[network.numLayers][network.maxAct]; // Initializes the activations 2-d array with proper dimensions

      /*
       * This loop fills the first layer of the activations array with values read from the input file.
       * The activation value are listed in order and immediately follow the "=" on each line.
       */
      for (int inputInd = 0; inputInd < network.numInputs; inputInd++)
      {
         String firstInputsLine = inputsReader.readLine();
         network.activations[0][inputInd] = Double.parseDouble(firstInputsLine.substring(firstInputsLine.indexOf("=") + 1));
      }
      String inputsLine3 = inputsReader.readLine();
      inputsLine3 += " ";
      int indexOfSpace = inputsLine3.indexOf(" ");
      network.Ti[0] = Double.parseDouble(inputsLine3.substring(inputsLine3.indexOf("=") + 1, indexOfSpace));
      for (int i = 1; i < network.numOutputs; i++)
      {
         network.Ti[i] = Double.parseDouble(inputsLine3.substring(indexOfSpace + 1, inputsLine3.indexOf(" ", indexOfSpace + 1)));
         indexOfSpace = inputsLine3.indexOf(" ", indexOfSpace);
      }
      String configLine3 = configReader.readLine();
      String configLine4 = configReader.readLine();

      network.weightsUpperBound = Double.parseDouble(configLine3.substring(configLine3.indexOf("=") + 1));
      network.weightsLowerBound = Double.parseDouble(configLine4.substring(configLine4.indexOf("=") + 1));
      String configLine5 = configReader.readLine();
      network.learningFactor = Double.parseDouble(configLine5.substring(configLine5.indexOf("=") + 1));
      String configLine6 = configReader.readLine();
      network.maxIterations = Integer.parseInt(configLine6.substring(configLine6.indexOf("=") + 1));
      String configLine7 = configReader.readLine();
      network.errorThreshold = Double.parseDouble(configLine7.substring(configLine7.indexOf("=") + 1));

      /*
       * Initializes a BufferedReader read input from weights file, or sets random weights if specified in args
       */
      if (weightsFile.equals("random"))
      {
         /*
          * Loops through the activations 2-d array and finds the maximum number of input, or left-hand activations in a layer
          * and the maximum number of output, or right-hand activations in a layer. All of the layers will be input layers
          * at some point when propagating through the network, since the final output layer is not included. However,
          * the first layer will never be an output layer, so its number of activations is excluded when calculating the maximum.
          *
          */
         for (int layerInd = 0; layerInd < network.numLayers; layerInd++)
         {
            if (layerInd < network.numLayers - 1 && network.activationDimensions[layerInd] > network.maxLeftActivations)
               network.maxLeftActivations = network.activationDimensions[layerInd];
            if (layerInd != 0 && network.activationDimensions[layerInd] > network.maxRightActivations)
               network.maxRightActivations = network.activationDimensions[layerInd];
         }

         network.weights = new double[network.numLayers][network.maxLeftActivations][network.maxRightActivations]; // The weights 3-d array is initialized

         /*
          * This loop fills the weights 3-d array by looping through all three dimensions. The outermost loop loops through
          * the number of layers. The next inner loop loops through the input, or left-hand side activations in the current layer.
          * The innermost loop loops through the output, or right-hand side activations, which are represented by the next layer.
          */
         for (int n = 0; n < network.numLayers - 1; n++)
         {
            for (int inputNode = 0; inputNode < network.activationDimensions[n]; inputNode++)
            {
               for (int outputNode = 0; outputNode < network.activationDimensions[n + 1]; outputNode++)
               {
                  network.weights[n][inputNode][outputNode] = Math.random() * (network.weightsUpperBound - network.weightsLowerBound) + network.weightsLowerBound;
               }
            }
         } // for (int n = 0; n < numLayers - 1; n++)
      }
      else
      {
         BufferedReader weightsReader;

         weightsReader = new BufferedReader(new FileReader(new File(weightsFile)));

         /*
          * Loops through the activations 2-d array and finds the maximum number of input, or left-hand activations in a layer
          * and the maximum number of output, or right-hand activations in a layer. All of the layers will be input layers
          * at some point when propagating through the network, since the final output layer is not included. However,
          * the first layer will never be an output layer, so its number of activations is excluded when calculating the maximum.
          *
          */
         for (int layerInd = 0; layerInd < network.numLayers; layerInd++)
         {
            if (layerInd < network.numLayers - 1 && network.activationDimensions[layerInd] > network.maxLeftActivations)
               network.maxLeftActivations = network.activationDimensions[layerInd];
            if (layerInd != 0 && network.activationDimensions[layerInd] > network.maxRightActivations)
               network.maxRightActivations = network.activationDimensions[layerInd];
         }

         network.weights = new double[network.numLayers][network.maxLeftActivations][network.maxRightActivations]; // The weights 3-d array is initialized

         /*
          * This loop fills the first layer of the weights 3-d array by looping through the input and hidden activations.
          * The outer loop loops through the input layer, or left-hand side activations.
          * The inner loop loops through the hidden layer, or right-hand side activations.
          */
         for (int k = 0; k < network.numInputs; k++)
         {
            for (int j = 0; j < network.numHidden; j++)
            {
               String line = weightsReader.readLine();
               network.weights[0][k][j] = Double.parseDouble(line.substring(line.indexOf("=") + 1));
            }
         }
         /*
          * This loop fills the second layer of the weights 3-d array by looping through the hidden and output activations.
          * The outer loop loops through the hidden layer, or left-hand side activations.
          * The inner loop loops through the output layer, or right-hand side activations.
          */
         for (int j = 0; j < network.numHidden; j++)
         {
            for (int i = 0; i < network.numOutputs; i++)
            {
               String line = weightsReader.readLine();
               network.weights[1][j][i] = Double.parseDouble(line.substring(line.indexOf("=") + 1));
            }
         }
      }
      network.hj = new double[network.numHidden];

      network.omegai = new double[network.numOutputs];

      network.thetai = new double[network.numOutputs];

      network.psii = new double[network.numOutputs];

      network.thetaj = new double[network.numHidden];

      network.omegaj = new double[network.numHidden];

      network.uppercasePsij = new double[network.numHidden];

      network.derivativeji = new double[network.numHidden][network.numOutputs];

      network.derivativekj = new double[network.numInputs][network.numHidden];

      network.deltaWeight = new double[network.numLayers - 1][network.maxLeftActivations][network.maxRightActivations];
      /*
       * Initializes a 2-d array representing the initial activations of the first training set.
       */
      double[][] firstSetActivations = new double[network.numLayers][network.maxAct];

      /*
       * Fills the 2-d array initialized above with activations set up by the network
       */
      for (int layer = 0; layer < network.numLayers; layer++)
      {
         for (int node = 0; node < network.activationDimensions[layer]; node++)
         {
            firstSetActivations[layer][node] = network.activations[layer][node];
         }
      }

      int trainingSetIndex = 0; // Keeps track of the current training set, whose activations are to be initialized

      /*
       * The following two lines initializes a 3-d array to store the activations of each training set and fills
       * the first index with the already initialized first set of activations.
       */
      double[][][] allSetsActivations = new double[numTrainingSets][network.numLayers][network.maxAct];

      allSetsActivations[trainingSetIndex] = firstSetActivations;

      /*
       * The following two lines initializes an array to store the truth values of each training set and fills the first
       * index with the already initialized truth value.
       */
      double[][] truthValues = new double[numTrainingSets][network.numOutputs];

      for (int i = 0; i < network.numOutputs; i++)
      {
         truthValues[0][i] = network.Ti[i];
      }

      trainingSetIndex++; // Increment's training set index since the first training set's activations and truth value has been initialized

      /*
       * Loops through all of the training set indices, resetting the activations arrays and truth values for each, then
       * storing the reset values into the allSetsActivations array.
       */
      while (trainingSetIndex < numTrainingSets)
      {
         for (int inputInd = 0; inputInd < network.numInputs; inputInd++)
         {
            String nextLine = inputsReader.readLine();
            network.activations[0][inputInd] = Double.parseDouble(nextLine.substring(nextLine.indexOf("=") + 1));
         }

         String thirdInputsLine = inputsReader.readLine();
         thirdInputsLine += " ";
         int spaceIndex = thirdInputsLine.indexOf(" ");
         network.Ti[0] = Double.parseDouble(thirdInputsLine.substring(thirdInputsLine.indexOf("=") + 1, spaceIndex));

         for (int i = 1; i < network.numOutputs; i++)
         {
            network.Ti[i] = Double.parseDouble(thirdInputsLine.substring(spaceIndex + 1, thirdInputsLine.indexOf(" ", spaceIndex + 1)));
            spaceIndex = thirdInputsLine.indexOf(" ", spaceIndex);
         }

         double[] currentSetTruthValues = network.Ti;

         for (int i = 0; i < network.numOutputs; i++)
         {
            truthValues[trainingSetIndex][i] = currentSetTruthValues[i];
         }

         for (int layer = 0; layer < network.numLayers; layer++)
         {
            for (int node = 0; node < network.activationDimensions[layer]; node++)
            {
               allSetsActivations[trainingSetIndex][layer][node] = network.activations[layer][node];
            }
         }
         trainingSetIndex++;
      } // while (trainingSetIndex < numTrainingSets)

      /*
       * Now, all weights, hyperparameters, activations, and truth values are initialized, so everything is printed out
       * before execution and training begins.
       */
      System.out.println("\n\nNetwork after initialization\n------------------------------");

      System.out.println("Learning factor: " + network.learningFactor);
      System.out.println("Max iterations: " + network.maxIterations);
      System.out.println("Error threshold: " + network.errorThreshold);
//      if (weightsFile.equals("random"))
//      {
         System.out.println("Weights random initialization range: " + network.weightsLowerBound + " to " + network.weightsUpperBound);
//      }
      String networkStructurePrintStatement = "Network Structure(layer lengths): ";
      for (int layer = 0; layer <network.numLayers; layer++)
      {
         /*
          * This loop prints out the layer lengths of the network by looping through activationDimensions, which
          * is set up in setActivations(BufferedReader configReader, BufferedReader inputsReader)
          */
         networkStructurePrintStatement += network.activationDimensions[layer] + " ";
      }
      System.out.println(networkStructurePrintStatement);

      System.out.println("Initial Weights: "); // Prints all initial weights

      for (int k = 0; k < network.numInputs; k++)
      {
         for (int j = 0; j < network.numHidden; j++)
         {
            System.out.println("w0" + k + j + "=" + network.weights[0][k][j]);
         }
      }

      for (int j = 0; j < network.numHidden; j++)
      {
         for (int i = 0; i < network.numOutputs; i++)
         {
            System.out.println("w1" + j + i + "=" + network.weights[1][j][i]);
         }
      }

      for (int trainingSetLoopIndex = 0; trainingSetLoopIndex < numTrainingSets; trainingSetLoopIndex++) // Prints all training sets' activations
      {
         String str = "Training set #" + (trainingSetLoopIndex + 1) + ", Truth values: ";

         for (int i = 0; i < network.numOutputs; i++)
         {
            str += truthValues[trainingSetLoopIndex][i] + ", ";
         }

         str += "Activations: " + Arrays.deepToString(allSetsActivations[trainingSetLoopIndex]);
         System.out.println(str);
      }

      /*
       * Now, the network is ready to begin execution and training.
       */
      double[][] errors = new double[numTrainingSets][network.numOutputs]; // Stores the errors of the set of weights when run on each training set

      double totalError = network.errorThreshold + 1; // Stores the sums of the errors of all training sets
      /*
       * Total error is initialized above with network.errorThreshold + 1 because it must be greater than the error threshold
       * of the network to start execution and training, and the actual total error has yet to be computed.
       * In other words, the network.errorThreshold + 1 is just a placeholder to allow execution to begin.
       * Once one round of execution is complete, the actual error will be calculated and used.
       */
      int iterationCounter = 0; // Counts how many iterations the network has trained for

      int trainingSetIndexForTraining = 0;   // Keeps track of which training set the network is training on

      /*
       * This loop runs until either the total error is lower than/equal to the predetermined error threshold or the network
       * has been training for the maximum number of iterations allowed.
       */
      while (totalError > network.errorThreshold && iterationCounter < network.maxIterations)
      {
         iterationCounter++; // Increments the counter of number of iterations before execution and training of each set
         trainingSetIndexForTraining = iterationCounter % numTrainingSets;

         network.activations = allSetsActivations[trainingSetIndexForTraining]; // Sets up the network with activations for this training set
         network.Ti = truthValues[trainingSetIndexForTraining];         // Sets up the network with truth values for this training set

         /*
          * First, before computing the output of the network for the current training set, all the activations that are not in the first
          * layer must be zeroed.
          */
         for (int n = 1; n < network.numLayers; n++)
         {
            for (int node = 0; node < network.activationDimensions[n]; node++)
            {
               network.activations[n][node] = 0.0;
            }
         }

         for (int n = 0; n < network.numLayers - 1; n++) // Loops through the number of layers that may possibly be inputting their values
         {
            for (int outputInd = 0; outputInd < network.activationDimensions[n + 1]; outputInd++) // Loops through each output activation
            {
               for (int inputInd = 0; inputInd < network.activationDimensions[n]; inputInd++)     // Loops through each input activation
               {
                  /*
                   * Adds the products of each activation and its corresponding weight to the
                   * current output activation.
                   */
                  network.activations[n + 1][outputInd] += network.activations[n][inputInd] * network.weights[n][inputInd][outputInd];
               }
               network.activations[n + 1][outputInd] = 1.0 / (1.0 + Math.exp(-network.activations[n + 1][outputInd] ));
            } // for (int outputInd = 0; outputInd < activationDimensions[n]; outputInd++)
         } // for (int n = 0; n < numLayers - 1; n++)

         for (int finalLayerInd = 0; finalLayerInd < network.numOutputs; finalLayerInd++) // Fills Fi with the last layer of activations
         {
            network.Fi[finalLayerInd] = network.activations[network.numLayers - 1][finalLayerInd];
         }

         for (int outputIndex = 0; outputIndex < network.numOutputs; outputIndex++)
         {
            network.omegai[outputIndex] = network.Ti[outputIndex] - network.Fi[outputIndex];
            network.Ei[outputIndex] = 0.5 * network.omegai[outputIndex] * network.omegai[outputIndex];
         }

         for (int j = 0; j < network.numHidden; j++)
         {
            /*
             * This loop zeros the hj's before summing for the new hj's of the current training set.
             */
            network.hj[j] = 0.0;
         }

         for (int j = 0; j < network.numHidden; j++)
         {
            double sum = 0.0;

            for (int k = 0; k < network.numInputs; k++)
            {
               sum += network.activations[0][k] * network.weights[0][k][j];
            }

            network.hj[j] = 1.0 / (1.0 + Math.exp(-sum));
         }

         for (int i = 0; i < network.numOutputs; i++)
         {
            /*
             * This loop zeros thetai's before summing for the new thetai's of the current training set.
             */
            network.thetai[i] = 0.0;
         }

         for (int i = 0; i < network.numOutputs; i++)
         {
            /*
             * This loop sums up the new thetai's of the current training set.
             */
            for (int j = 0; j < network.numHidden; j++)
            {
               network.thetai[i] += network.hj[j] * network.weights[1][j][i];
            }
         }

         for (int i = 0; i < network.numOutputs; i++)
         {
            network.Fi[i] = 1.0 / (1.0 + Math.exp(-network.thetai[i])); // Calculates Fi's for the current training set
         }

         for (int i = 0; i < network.numOutputs; i++)
         {
            network.omegai[i] = network.Ti[i] - network.Fi[i]; // Calculates omegai's for the current training set
         }

         for (int i = 0; i < network.numOutputs; i++)
         {
            double activation = 1.0 / (1.0 + Math.exp(-network.thetai[i]));
            double thresDer = activation * (1.0 - activation);
            network.psii[i] = network.omegai[i] * thresDer; // Calculates psii's for the current training set
         }

         /*
          * This loop zeros thetaj before summing the new thetaj's for this iteration of training.
          */
         for (int j = 0; j < network.numHidden; j++)
         {
            network.thetaj[j] = 0.0;
         }

         /*
          * This loop calculates thetaj for the current training set.
          */
         for (int j = 0; j < network.numHidden; j++)
         {
            for (int k = 0; k < network.numInputs; k++)
            {
               network.thetaj[j] += network.activations[0][k] * network.weights[0][k][j];
            }
         }

         /*
          * This loop zeros omegaj before calculating the sums.
          */
         for (int j = 0; j < network.numHidden; j++)
         {
            network.omegaj[j] = 0.0;
         }

         /*
          * This loop calculates the omegaj's for the current training set.
          */
         for (int j = 0; j < network.numHidden; j++)
         {
            for (int i = 0; i < network.numOutputs; i++)
            {
               network.omegaj[j] += network.psii[i] * network.weights[1][j][i];
            }
         }

         /*
          * This loop calculates the uppercasePsi's for the current training set.
          */
         for (int j = 0; j < network.numHidden; j++)
         {
            double activation = 1.0 / (1.0 + Math.exp(-network.thetaj[j]));
            double thresholdDerivative = activation * (1.0 - activation);
            network.uppercasePsij[j] = network.omegaj[j] * thresholdDerivative;
         }

         /*
          * This loop fills the first layer of the derivativekj array according to values calculated above for this training set.
          */
         for (int k = 0; k < network.numInputs; k++)
         {
            for (int j = 0; j < network.numHidden; j++)
            {
               network.derivativekj[k][j] = -network.activations[0][k] * network.uppercasePsij[j];
            }
         }

         /*
          * This loop fills the first layer of the derivativeji array according to values calculated above for this training set.
          */
         for (int j = 0; j < network.numHidden; j++)
         {
            for (int i = 0; i < network.numOutputs; i++)
            {
               network.derivativeji[j][i] = -network.hj[j] * network.psii[i];
            }
         }

         /*
          * This loop calculates the delta weights for the first layer of weights of the current training set
          */
         for (int k = 0; k < network.numInputs; k++)
         {
            for (int j = 0; j < network.numHidden; j++)
            {
               network.deltaWeight[0][k][j] = -network.learningFactor * network.derivativekj[k][j];
            }
         }

         /*
          * This loop calculates the delta weights for the second layer of weights of the current training set
          */
         for (int j = 0; j < network.numHidden; j++)
         {
            for (int i = 0; i < network.numOutputs; i++)
            {
               network.deltaWeight[1][j][i] = -network.learningFactor * network.derivativeji[j][i];
            }
         }

         /*
          * This loop modifies the the weights array based on the deltaWeights calculated above.
          */
         for (int alpha = 0; alpha < network.numLayers - 1; alpha++)
         {
            for (int beta = 0; beta < network.activationDimensions[alpha]; beta++)
            {
               for (int gamma = 0; gamma < network.activationDimensions[alpha + 1]; gamma++)
               {
                  network.weights[alpha][beta][gamma] += network.deltaWeight[alpha][gamma][beta];
               }
            }
         }

         for (int i = 0; i < network.numOutputs; i++)
         {
            errors[trainingSetIndexForTraining][i] = network.Ei[i];   // Updates errors array
         }

         totalError = 0.0; // Zeros the sum of the errors and then re-sums up errors for the total error

         for (double[] errorSet : errors)
         {
            for (double error: errorSet)
            {
               totalError += error;
            }
         }
      } // while (totalError > network.errorThreshold && iterationCounter < network.getMaxIterations())
      /*
       * Now, the network should have either 1) determined the correct weights to compute outputs for all training sets with
       * an error lower than the given threshold or 2) reached the maximum number of iterations of training allowed.
       * At this point, the results of training can be printed in the console for the user to review.
       */
      System.out.println("\n\nNetwork after training\n------------------------------");
      System.out.println("Final Weights: "); // Prints all weights after training
      for (int k = 0; k < network.numInputs; k++)
      {
         for (int j = 0; j < network.numHidden; j++)
         {
            System.out.println("w0" + k + j + "=" + network.weights[0][k][j]);
         }
      }

      for (int j = 0; j < network.numHidden; j++)
      {
         for (int i = 0; i < network.numOutputs; i++)
         {
            System.out.println("w1" + j + i + "=" + network.weights[1][j][i]);
         }
      }

      System.out.println("Total error: " + totalError);                 // Prints the total error for execution of each training set
      System.out.println("Number of iterations: " + iterationCounter);  // Prints out the number of iterations the network trained for

      for (int trainingSetLoopIndex = 0; trainingSetLoopIndex < numTrainingSets; trainingSetLoopIndex++)
      {
         /*
          * The next three lines executes the network using the final, determined weights and activations corresponding
          * to the current training set.
          */
         network.activations = allSetsActivations[trainingSetLoopIndex];

         network.Ti = truthValues[trainingSetLoopIndex];

         /*
          * First, before computing the output of the network for the current training set, all the activations that are not in the first
          * layer must be zeroed.
          */
         for (int n = 1; n < network.numLayers; n++)
         {
            for (int node = 0; node < network.activationDimensions[n]; node++)
            {
               network.activations[n][node] = 0.0;
            }
         }

         for (int n = 0; n < network.numLayers - 1; n++) // Loops through the number of layers that may possibly be inputting their values
         {
            for (int outputInd = 0; outputInd < network.activationDimensions[n + 1]; outputInd++) // Loops through each output activation
            {
               for (int inputInd = 0; inputInd < network.activationDimensions[n]; inputInd++)     // Loops through each input activation
               {
                  /*
                   * Adds the products of each activation and its corresponding weight to the
                   * current output activation.
                   */
                  network.activations[n + 1][outputInd] += network.activations[n][inputInd] * network.weights[n][inputInd][outputInd];
               }
               network.activations[n + 1][outputInd] = 1.0 / (1.0 + Math.exp(-network.activations[n + 1][outputInd])); // Passes sum into threshold function
            } // for (int outputInd = 0; outputInd < activationDimensions[n]; outputInd++)
         } // for (int n = 0; n < numLayers - 1; n++)

         for (int finalLayerInd = 0; finalLayerInd < network.numOutputs; finalLayerInd++) // Fills Fi with the last layer of activations
         {
            network.Fi[finalLayerInd] = network.activations[network.numLayers - 1][finalLayerInd];
         }

         for (int outputIndex = 0; outputIndex < network.numOutputs; outputIndex++)
         {
            network.omegai[outputIndex] = network.Ti[outputIndex] - network.Fi[outputIndex];
            network.Ei[outputIndex] = 0.5 * network.omegai[outputIndex] * network.omegai[outputIndex];
         }
         /*
          * Prints out, for each training set, what output was computed, the corresponding truth value,
          * and the error of the output.
          */
         String ret = "";

         for (double output : network.Fi)
         {
            // Loops through the Ei array
            ret += output + ", ";
         }

         String str = "Training set #" + (trainingSetLoopIndex + 1) + ", Outputs: " + ret.substring(0, ret.length() - 1)
               + " Truth values: ";

         for (int i = 0; i < network.numOutputs; i++)
         {
            str += truthValues[trainingSetLoopIndex][i] + ", ";
         }

         str += "Error: ";
         String ret2 = "";

         for (double error : network.Ei)
         {
            // Loops through the Ei array
            ret2 += error + ", ";
         }
         str += ret2.substring(0, ret2.length() - 1);
         System.out.println(str.substring(0, str.length() - 1));
      }
   } // public static void main(String[] args) throws IOException, FileNotFoundException
} // public class Network