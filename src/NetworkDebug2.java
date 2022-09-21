import java.io.*;

/*
 * @author Dawson Chen
 * @since 2.18.2020
 * The NetworkDebug2 class implements a neural network with 1 hidden layer, which allows the user to determine the number of activations
 * in the input, hidden, and output layers. It performs gradient descent given hyperparameters that are specified by the user,
 * such as the rate at which it trains, the range in which weights should be initialized randomly, and when to stop training.
 * This assignment is done for Dr. Nelson's AT Neural Network class (P2).
 */
public class NetworkDebug2
{
   /*
    * This is a 2d array of doubles that stores all of the network's activations for the current training set.
    * The first dimension represents layers, and the second dimension represents the activations in a layer.
    */
   private double[][] activations;

   /*
    * This is a 3d array of doubles that stores all of the network's weights across all training sets.
    * The first dimension represents layers. The second dimension represents the activations in the input, or left-hand side layer.
    * The third dimension represents the output, or right-hand side layer.
    */
   private double[][][] weights;

   private double[] Ti; // Stores the truth values of the current training set
   private double[] Ei; // Stores the errors of the current training set with respect to Ti

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
    * This method prints the learning factor, the maximum number of iterations allowed, and the error threshold into the console.
    * @param randomWeights whether weights were randomly initialized
    */
   public void printHyperparameters(boolean randomWeights)
   {
      System.out.println("Learning factor: " + learningFactor);
      System.out.println("Max iterations: " + maxIterations);
      System.out.println("Error threshold: " + errorThreshold);

      if (randomWeights)
      {
         System.out.println("Weights random initialization range: " + weightsLowerBound + " to " + weightsUpperBound);
      }

      String networkStructurePrintStatement = "Network Structure(layer lengths): ";

      for (int layer = 0; layer < numLayers; layer++)
      {
         networkStructurePrintStatement += activationDimensions[layer] + " "; // Assembles the activation dimensions in one printed line
      }
      System.out.println(networkStructurePrintStatement);
   }

   /**
    * This method directly sets the activations 2-d array to an inputted parameter.
    * @param activations is the array whose values are to be copied into this network's activations array directly
    */
   public void setActivations(double[][] activations)
   {
      for (int n = 0; n < numLayers; n++)
      {
         for (int node = 0; node < activationDimensions[n]; node++)
         {
            this.activations[n][node] = activations[n][node];
         }
      }
   }

   /**
    * This method directly sets the truth value to an inputted parameter.
    * @param truthValues
    */
   public void setTruthValues(double[] truthValues)
   {
      for (int i = 0; i < numOutputs; i++)
      {
         this.Ti[i] = truthValues[i];
      }
   }

   /**
    * This method sets up the arrays that will be needed in training the network. They are initialized here, only once ever,
    * so as to never allocate extra memory during the training loop itself. They will simply be zeroed for each new iteration.
    */
   public void setTrainingVariables()
   {
      omegai = new double[numOutputs];

      thetai = new double[numOutputs];

      psii = new double[numOutputs];

      thetaj = new double[numHidden];

      omegaj = new double[numHidden];

      uppercasePsij = new double[numHidden];

      derivativeji = new double[numHidden][numOutputs];

      derivativekj = new double[numInputs][numHidden];

      deltaWeight = new double[numLayers - 1][maxLeftActivations][maxRightActivations];
   } // public void setTrainingVariables()

   /**
    * This method reads in the structure of the network from a configuration file and the values of the input layer
    * from an input file. With this information, this method initializes the activations array with appropriate
    * dimensions and populates the input (0th) layer with values of the first training set. This method is only called
    * once in the training process, as it manages only the initial setup of the first training set.
    *
    * @param configReader is a BufferedReader of the configuration file containing the structure of the network.
    * @param inputsReader is a BufferedReader of the inputs file containing the values in the input layer.
    * @throws IOException if there are no more lines to read in the inputs or config file.
    */
   public void setActivations(BufferedReader configReader, BufferedReader inputsReader) throws IOException
   {
      /*
       * The length of the first dimension of the activations array is the number of layers,
       * and the length the second dimension is the maximum number of nodes in a layer. The maximum number of nodes
       * are used rather than the exact number of nodes because it is impossible to initialize an array in java
       * with different lengths in the second dimension.
       */
      String configLine1 = configReader.readLine();                                       // Reads the first line of the config file
      numLayers = Integer.parseInt(configLine1.substring(configLine1.indexOf("=") + 1));  // Finds number of layers of activations
      /*
       * Note: the number of layers inputted by the user refers to the number of layers of weights, therefore it does not account for
       * the output layer.
       */
      activationDimensions = new int[numLayers]; // Represents the lengths of each layer

      String line2 = configReader.readLine();    // Reads the second line of the configuration file
      int index = line2.indexOf("=") + 1;        // Numbers after this "=" are layer lengths

      line2 += " "; // Adjusts the line to make layer lengths more easily decipherable when using indexOf to find the next space
      /*
       * This loop fills activationDimensions with inputted lengths of each layer from the configuration file.
       * The length of each layer is separated by white spaces.
       */
      for (int layer = 0; layer < numLayers; layer++)
      {
         activationDimensions[layer] = Integer.parseInt(line2.substring(index, line2.indexOf(" ", index)));
         index = line2.indexOf(" ", index + 1) + 1;
      }

      numInputs = activationDimensions[0]; // Sets the number of inputs to the first layer's activation dimensions

      numHidden = activationDimensions[1]; // Sets the number of hidden activations to be the second layer's activation dimensions

      numOutputs = activationDimensions[numLayers - 1]; // Sets the number of outputs to the last layer's activation dimensions

      Ti = new double[numOutputs];

      Ei = new double[numOutputs];

      maxAct = 0; // Represents the maximum number of activations in a layer in activations, finds it by looping through all
      for (int numActs : activationDimensions)
      {
         if (numActs > maxAct)
            maxAct = numActs;
      }
      this.activations = new double[numLayers][maxAct]; // Initializes the activations 2-d array with proper dimensions

      /*
       * This loop fills the first layer of the activations array with values read from the input file.
       * The activation value are listed in order and immediately follow the "=" on each line.
       */
      for (int inputInd = 0; inputInd < numInputs; inputInd++)
      {
         String inputsLine1 = inputsReader.readLine();
         activations[0][inputInd] = Double.parseDouble(inputsLine1.substring(inputsLine1.indexOf("=") + 1));
      }
   } // public void setActivations(BufferedReader configReader, BufferedReader inputsReader) throws IOException

   /**
    * This method continues reading input from the inputsReader to collect values of the next training set.
    * @param inputsReader is a BufferedReader of the inputs file containing the values in the input layer.
    * @throws IOException if there are no more lines to read in the inputs file.
    */
   public void resetActivations(BufferedReader inputsReader) throws IOException
   {
      /*
       * This loop fills the first layer of the activations array with values read from the input file.
       * The activation value are listed in order and immediately follow the "=" on each line.
       */
      for (int inputInd = 0; inputInd < numInputs; inputInd++)
      {
         String nextLine = inputsReader.readLine();
         activations[0][inputInd] = Double.parseDouble(nextLine.substring(nextLine.indexOf("=") + 1));
      }
   }

   /**
    * This method reads the number of training sets from the inputsReader, listed on the first line of the inputs file.
    * @param inputsReader is a BufferedReader of the inputs file containing the values in the input layer.
    * @return the number of trainings sets listed in the inputs file
    * @throws IOException
    */
   public int getNumTrainingSets(BufferedReader inputsReader) throws IOException
   {
      String inputsLine1 = inputsReader.readLine();
      return Integer.parseInt(inputsLine1.substring(inputsLine1.indexOf("=") + 1));
   }

   /**
    * This method initializes random weights according to the known structure, given in the configuration file.
    * The length of the first dimension in the weights 3-d array is the number of layers. The length of the second
    * dimension is the maximum length of any input, or left-hand side layer of activations. The length of the third
    * dimension is the maximum length of the output, or right-hand side layer of activations.
    * @param lowerBound is the lower bound for initializing random weights
    * @param upperBound is the upper bound for initializing random weights
    */
   public void setRandomWeights(double lowerBound, double upperBound)
   {
      /*
       * Loops through the activations 2-d array and finds the maximum number of input, or left-hand activations in a layer
       * and the maximum number of output, or right-hand activations in a layer. All of the layers will be input layers
       * at some point when propagating through the network, since the final output layer is not included. However,
       * the first layer will never be an output layer, so its number of activations is excluded when calculating the maximum.
       *
       */
      for (int layerInd = 0; layerInd < numLayers; layerInd++)
      {
         if (layerInd < numLayers - 1 && activationDimensions[layerInd] > maxLeftActivations)
            maxLeftActivations = activationDimensions[layerInd];
         if (layerInd != 0 && activationDimensions[layerInd] > maxRightActivations)
            maxRightActivations = activationDimensions[layerInd];
      }

      weights = new double[numLayers][maxLeftActivations][maxRightActivations]; // The weights 3-d array is initialized

      /*
       * This loop fills the weights 3-d array by looping through all three dimensions. The outermost loop loops through
       * the number of layers. The next inner loop loops through the input, or left-hand side activations in the current layer.
       * The innermost loop loops through the output, or right-hand side activations, which are represented by the next layer.
       */
      for (int n = 0; n < numLayers - 1; n++)
      {
         for (int inputNode = 0; inputNode < activationDimensions[n]; inputNode++)
         {
            for (int outputNode = 0; outputNode < activationDimensions[n + 1]; outputNode++)
            {
               weights[n][inputNode][outputNode] = generateRandomNumber(lowerBound, upperBound);
            }
         }
      } // for (int n = 0; n < numLayers - 1; n++)
   } // public void setRandomWeights(double lowerBound, double upperBound)

   /**
    * Generates a random double in the range [lowerBound, upperBound)
    * @param lowerBound the lower bound of the randomization
    * @param upperBound the upper bound of the randomization
    * @return
    */
   public double generateRandomNumber(double lowerBound, double upperBound)
   {
      return Math.random() * (upperBound - lowerBound) + lowerBound;
   }

   /**
    * This method reads in the values of the weights in the network, and with an already initialized activations array,
    * initializes the weights array with appropriate dimensions and populates it according to the values specified in
    * the file. The length of the first dimension in the weights 3-d array is the number of layers. The length of the second
    * dimension is the maximum length of any input, or left-hand side layer of activations. The length of the third
    * dimension is the maximum length of the output, or right-hand side layer of activations.
    *
    * @param weightsReader is a BufferedReader of the weights file containing the values of all of the weights
    * @throws IOException if there are no more lines to read in the weights file.
    * @precondition the activations array has been initialized with appropriate dimensions.
    */
   public void setWeights(BufferedReader weightsReader) throws IOException
   {
      /*
       * Loops through the activations 2-d array and finds the maximum number of input, or left-hand activations in a layer
       * and the maximum number of output, or right-hand activations in a layer. All of the layers will be input layers
       * at some point when propagating through the network, since the final output layer is not included. However,
       * the first layer will never be an output layer, so its number of activations is excluded when calculating the maximum.
       *
       */
      for (int layerInd = 0; layerInd < numLayers; layerInd++)
      {
         if (layerInd < numLayers - 1 && activationDimensions[layerInd] > maxLeftActivations)
            maxLeftActivations = activationDimensions[layerInd];
         if (layerInd != 0 && activationDimensions[layerInd] > maxRightActivations)
            maxRightActivations = activationDimensions[layerInd];
      }

      weights = new double[numLayers][maxLeftActivations][maxRightActivations]; // The weights 3-d array is initialized

      /*
       * This loop fills the first layer of the weights 3-d array by looping through the input and hidden activations.
       * The outer loop loops through the input layer, or left-hand side activations.
       * The inner loop loops through the hidden layer, or right-hand side activations.
       */
      for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            String line = weightsReader.readLine();
            weights[0][k][j] = Double.parseDouble(line.substring(line.indexOf("=") + 1));
         }
      }
      /*
       * This loop fills the second layer of the weights 3-d array by looping through the hidden and output activations.
       * The outer loop loops through the hidden layer, or left-hand side activations.
       * The inner loop loops through the output layer, or right-hand side activations.
       */
      for (int j = 0; j < numHidden; j++)
      {
         for (int i = 0; i < numOutputs; i++)
         {
            String line = weightsReader.readLine();
            weights[1][j][i] = Double.parseDouble(line.substring(line.indexOf("=") + 1));
         }
      }
   } // public void setWeights(BufferedReader weightsReader) throws IOException

   /**
    * This method reads in the weights bounds from the configReader and stores it.
    * @param configReader is a BufferedReader containing the weights bounds on the next lines
    * @throws IOException if there are no more lines to read in the config file.
    */
   private void setWeightsBounds(BufferedReader configReader) throws IOException
   {
      String configLine3 = configReader.readLine();
      String configLine4 = configReader.readLine();

      weightsUpperBound = Double.parseDouble(configLine3.substring(configLine3.indexOf("=") + 1));
      weightsLowerBound = Double.parseDouble(configLine4.substring(configLine4.indexOf("=") + 1));
   }

   /**
    * This method reads in the truth values of the current training set from the inputsReader and stores them.
    * @param inputsReader is a BufferedReader that contains the truth values.
    * @return the truth values
    * @throws IOException if there are no more lines to read in the inputs file.
    */
   public void setTruthValues(BufferedReader inputsReader) throws IOException
   {
      String inputsLine3 = inputsReader.readLine();
      inputsLine3 += " ";
      int indexOfSpace = inputsLine3.indexOf(" ");
      Ti[0] = Double.parseDouble(inputsLine3.substring(inputsLine3.indexOf("=") + 1, indexOfSpace));
      for (int i = 1; i < numOutputs; i++)
      {
         Ti[i] = Double.parseDouble(inputsLine3.substring(indexOfSpace + 1, inputsLine3.indexOf(" ", indexOfSpace + 1)));
         indexOfSpace = inputsLine3.indexOf(" ", indexOfSpace + 1);
      }
   } // public void setTruthValues(BufferedReader inputsReader) throws IOException

   /**
    * This method reads in the learning factor from a configReader and stores it.
    * @param configReader is a BufferedReader containing the learning factor on the next line
    * @throws IOException if there are no more lines to read in the config file.
    */
   public void setLearningFactor(BufferedReader configReader) throws IOException
   {
      String configLine5 = configReader.readLine();
      learningFactor = Double.parseDouble(configLine5.substring(configLine5.indexOf("=") + 1));
   }

   /**
    * This method reads in the maximum number of iterations from a configReader and stores it.
    * @param configReader is a BufferedReader containing the maximum number of iterations on the next line
    * @throws IOException if there are no more lines to read in the config file.
    */
   public void setMaxIterations(BufferedReader configReader) throws IOException
   {
      String configLine6 = configReader.readLine();
      maxIterations = Integer.parseInt(configLine6.substring(configLine6.indexOf("=") + 1));
   }

   /**
    * This method reads in the error threshold from a configReader and stores it.
    * @param configReader is a BufferedReader containing the error threshold on the next line
    * @throws IOException if there are no more lines to read in the config file.
    */
   public void setErrorThreshold(BufferedReader configReader) throws IOException
   {
      String configLine7 = configReader.readLine();
      errorThreshold = Double.parseDouble(configLine7.substring(configLine7.indexOf("=") + 1));
   }

   /**
    * This method modifies a value by passing it into a threshold function, which is currently a sigmoid.
    * Sigmoid: f(x) = 1.0 / (1.0 + exp(-x))
    * @return the output after passed into the threshold function
    */
   public double thresholdFunction(double value)
   {
      return 1.0 / (1.0 + Math.exp(-value));
   }

   /**
    * This method computes the derivative of the threshold function.
    * @return the derivative of the threshold function
    */
   public double thresholdDerivative(double value)
   {
      double activation = thresholdFunction(value);
      return activation * (1.0 - activation);
   }

   /**
    * This method sets up the network with a configuration, weights, and inputs that the user specifies.
    * This method is called by the main method, so it is possible for the main method to pass input from the user from the
    * command line to this method, in the form of a String array called args.
    *
    * @param network the network to be set up
    * @param args    the arguments passed by the user through the command line, into the main method
    * @throws IOException if the input files have unexpected formats
    * @return the number of training sets that this network will have
    */
   public static int setup(NetworkDebug2 network, String[] args) throws IOException
   {
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

      int numTrainingSets = network.getNumTrainingSets(inputsReader);

      network.setActivations(configReader, inputsReader);
      network.setTruthValues(inputsReader);
      network.setWeightsBounds(configReader);
      network.setLearningFactor(configReader);
      network.setMaxIterations(configReader);
      network.setErrorThreshold(configReader);

      /*
       * Initializes a BufferedReader read input from weights file, or sets random weights if specified in args
       */
      if (weightsFile.equals("random"))
      {
         network.setRandomWeights(network.weightsUpperBound, network.weightsLowerBound);
      }
      else
      {
         BufferedReader weightsReader;

         weightsReader = new BufferedReader(new FileReader(new File(weightsFile)));

         network.setWeights(weightsReader);
      }
      network.setTrainingVariables(); // Initializes all variables needed for training
      return numTrainingSets;
   } // public static int setup(NetworkDebug2 network, String[] args) throws IOException

   /**
    * This method updates each activation in each layer of the network by multiplying each activation in an input, or
    * left-hand side layer with corresponding weights and passing the sum of the products through a threshold function.
    * The output of the threshold function is the new, updated value of an activation in the output, or right-hand side
    * layer. The final output layer's updated activation is the output of the network. Throughout its execution, this method
    * modifies the activations array.
    */
   public void computeOutput()
   {
      for (int n = 0; n < numLayers - 1; n++) // Loops through the number of layers that may possibly be inputting their values
      {
         for (int outputInd = 0; outputInd < activationDimensions[n + 1]; outputInd++) // Loops through each output activation
         {
            for (int inputInd = 0; inputInd < activationDimensions[n]; inputInd++)     // Loops through each input activation
            {
               /*
                * Adds the products of each activation and its corresponding weight to the
                * current output activation.
                */
               activations[n + 1][outputInd] += activations[n][inputInd] * weights[n][inputInd][outputInd];
            }
            activations[n + 1][outputInd] = thresholdFunction(activations[n + 1][outputInd]); // Passes sum into threshold function
         } // for (int outputInd = 0; outputInd < activationDimensions[n]; outputInd++)
      } // for (int n = 0; n < numLayers - 1; n++)
   }

   /**
    * This method computes the error of the current training set's final output activations in comparison to the truth values given by the input file.
    */
   public void computeErrors()
   {
      /*
       * Calculates all the errors for each output of the network
       */
      for (int outputIndex = 0; outputIndex < numOutputs; outputIndex++)
      {
         omegai[outputIndex] = Ti[outputIndex] - activations[2][outputIndex];
         Ei[outputIndex] = 0.5 * omegai[outputIndex] * omegai[outputIndex];
      }
   }

   /**
    * This method computes how much to modify each weight in order to perform gradient descent and make the network
    * produce the most accurate output. This method takes into account the current training set's activations only, but
    * the network will be trained across multiple training sets by running this method many times. All the variables have
    * names corresponding to the description in Dr. Nelson's design document,
    * "Minimization of the Error Function for a Single Output and One Hidden Layer". The design document describes how
    * each of the variables below contributes to calculating how to change weights for gradient descent and their
    * mathematical relationship to one another.
    */
   public void train()
   {
      for (int j = 0; j < numHidden; j++)
      {
         /*
          * This loop zeros the second layer activation's before summing for the new second layer activation's of the current training set.
          */
         activations[1][j] = 0.0;
      }

      for (int j = 0; j < numHidden; j++)
      {
         /*
          * This loop sums up the values in activations
          */
         double sum = 0.0;
         for (int k = 0; k < numInputs; k++)
         {
            sum += activations[0][k] * weights[0][k][j];
         }
         activations[1][j] = thresholdFunction(sum);
      }

      for (int i = 0; i < numOutputs; i++)
      {
         /*
          * This loop zeros thetai's before summing for the new thetai's of the current training set.
          */
         thetai[i] = 0.0;
      }

      for (int i = 0; i < numOutputs; i++)
      {
         /*
          * This loop sums up the new thetai's of the current training set.
          */
         for (int j = 0; j < numHidden; j++)
         {
            thetai[i] += activations[1][j] * weights[1][j][i];
         }
      }

      for (int i = 0; i < numOutputs; i++)
      {
         activations[2][i] = thresholdFunction(thetai[i]); // Calculates the final layer's for the current training set
      }

      for (int i = 0; i < numOutputs; i++)
      {
         omegai[i] = Ti[i] - activations[2][i]; // Calculates omegai's for the current training set
      }

      for (int i = 0; i < numOutputs; i++)
      {
         psii[i] = omegai[i] * thresholdDerivative(thetai[i]); // Calculates psii's for the current training set
      }

      /*
       * This loop zeros thetaj before summing the new thetaj's for this iteration of training.
       */
      for (int j = 0; j < numHidden; j++)
      {
         thetaj[j] = 0.0;
      }

      /*
       * This loop calculates thetaj for the current training set.
       */
      for (int j = 0; j < numHidden; j++)
      {
         for (int k = 0; k < numInputs; k++)
         {
            thetaj[j] += activations[0][k] * weights[0][k][j];
         }
      }

      /*
       * This loop zeros omegaj before calculating the sums.
       */
      for (int j = 0; j < numHidden; j++)
      {
         omegaj[j] = 0.0;
      }

      /*
       * This loop calculates the omegaj's for the current training set.
       */
      for (int j = 0; j < numHidden; j++)
      {
         for (int i = 0; i < numOutputs; i++)
         {
            omegaj[j] += psii[i] * weights[1][j][i];
         }
      }

      /*
       * This loop calculates the uppercasePsi's for the current training set.
       */
      for (int j = 0; j < numHidden; j++)
      {
         uppercasePsij[j] = omegaj[j] * thresholdDerivative(thetaj[j]);
      }

      /*
       * This loop fills the first layer of the derivativekj array according to values calculated above for this training set.
       */
      for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            derivativekj[k][j] = -activations[0][k] * uppercasePsij[j];
         }
      }

      /*
       * This loop fills the first layer of the derivativeji array according to values calculated above for this training set.
       */
      for (int j = 0; j < numHidden; j++)
      {
         for (int i = 0; i < numOutputs; i++)
         {
            derivativeji[j][i] = -activations[1][j] * psii[i];
         }
      }

      /*
       * This loop calculates the delta weights for the first layer of weights of the current training set
       */
      for (int k = 0; k < numInputs; k++)
      {
         for (int j = 0; j < numHidden; j++)
         {
            deltaWeight[0][k][j] = -learningFactor * derivativekj[k][j];
         }
      }

      /*
       * This loop calculates the delta weights for the second layer of weights of the current training set
       */
      for (int j = 0; j < numHidden; j++)
      {
         for (int i = 0; i < numOutputs; i++)
         {
            deltaWeight[1][j][i] = -learningFactor * derivativeji[j][i];
         }
      }

      /*
       * This loop modifies the the weights array based on the deltaWeights calculated above.
       */
      for (int alpha = 0; alpha < numLayers - 1; alpha++)
      {
         for (int beta = 0; beta < activationDimensions[alpha]; beta++)
         {
            for (int gamma = 0; gamma < activationDimensions[alpha + 1]; gamma++)
            {
               weights[alpha][beta][gamma] += deltaWeight[alpha][beta][gamma];
            }
         }
      }
   } // public void train()

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
      NetworkDebug2 network = new NetworkDebug2(); // Initializes a Network object, on which following methods are run

      /*
       * The following line sets up the hyperparameters and weights of the network. It also initializes the first training set's
       * activations and truth value.
       */
      int numTrainingSets = setup(network, args);

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

      for (int n = 0; n < network.numLayers; n++)
      {
         for (int node = 0; node < network.activationDimensions[n]; node++)
         {
            allSetsActivations[trainingSetIndex][n][node] = firstSetActivations[n][node];
         }
      }

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
         network.resetActivations(inputsReader);
         network.setTruthValues(inputsReader);

         for (int i = 0; i < network.numOutputs; i++)
         {
            truthValues[trainingSetIndex][i] = network.Ti[i];
         }
         for (int n = 0; n < network.numLayers; n++)
         {
            for (int node = 0; node < network.activationDimensions[n]; node++)
            {
               allSetsActivations[trainingSetIndex][n][node] = network.activations[n][node];
            }
         }
         trainingSetIndex++;
      } // while (trainingSetIndex < numTrainingSets)

      /*
       * Now, all weights, hyperparameters, activations, and truth values are initialized, so everything is printed out
       * before execution and training begins.
       */
      System.out.println("\n\nNetwork after initialization\n------------------------------");

      network.printHyperparameters(args[1].equals("random")); // Prints hyperparameters

      System.out.println("Initial Weights: ");

      /*
       * The following two loops prints out all initial weights
       */

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

      /*
       * This loop  prints out the initial activations and the truth table.
       */
      for (int trainingSetLoopIndex = 0; trainingSetLoopIndex < numTrainingSets; trainingSetLoopIndex++)
      {
         String str = "Training set #" + (trainingSetLoopIndex + 1) + ", Truth values: ";
         for (int i = 0; i < network.numOutputs; i++)
         {
            str += truthValues[trainingSetLoopIndex][i] + ", ";
         }
         for (int n = 0; n < network.numLayers; n++)
         {
            str += "Layer " + n + " activations: ";
            for (int node = 0; node < network.activationDimensions[n]; node++)
            {
               str += allSetsActivations[trainingSetLoopIndex][n][node] + " ";
            }
         }
         System.out.println(str);
      }  // for (int trainingSetLoopIndex = 0; trainingSetLoopIndex < numTrainingSets; trainingSetLoopIndex++)

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

         network.setActivations(allSetsActivations[trainingSetIndexForTraining]); // Sets up the network with activations for this training set
         network.setTruthValues(truthValues[trainingSetIndexForTraining]);        // Sets up the network with truth values for this training set

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

         network.computeOutput();
         network.computeErrors();

         network.train();

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

         totalError /= (double)(network.numOutputs); // Makes sure that the total error is independent of the number of output activations
         totalError /= (double)(numTrainingSets);    // Makes sure that the total error is independent of the number of training sets used

      } // while (totalError > network.errorThreshold && iterationCounter < network.getMaxIterations())
      /*
       * Now, the network should have either 1) determined the correct weights to compute outputs for all training sets with
       * an error lower than the given threshold or 2) reached the maximum number of iterations of training allowed.
       * At this point, the results of training can be printed in the console for the user to review.
       */
      System.out.println("\n\nNetwork after training\n------------------------------");
      System.out.println("Final Weights: "); // Prints all weights after training

      for (int n = 0; n < network.numLayers - 1; n++)
      {
         for (int inputInd = 0; inputInd < network.activationDimensions[n]; inputInd++)
         {
            for (int outputInd = 0; outputInd < network.activationDimensions[n+1]; outputInd++)
            {
               System.out.println("w" + n + inputInd + outputInd + "=" + network.weights[n][inputInd][outputInd]);
            }
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
         network.setActivations(allSetsActivations[trainingSetLoopIndex]);
         network.setTruthValues(truthValues[trainingSetLoopIndex]);

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

         network.computeOutput();
         network.computeErrors();

         /*
          * Prints out, for each training set, what output was computed, the corresponding truth value,
          * and the error of the output.
          */
         String str = "Training set #" + (trainingSetLoopIndex + 1) + ", Outputs: ";
         for (int i = 0; i < network.numOutputs; i++)
         {
            str += network.activations[2][i] + ", ";
         }

         str += " Truth values: ";
         for (int i = 0; i < network.numOutputs; i++)
         {
            str += truthValues[trainingSetLoopIndex][i] + ", ";
         }

         str += "Errors: ";
         for (int i = 0; i < network.numOutputs; i++)
         {
            str += network.Ei[i] + " ";
         }

         System.out.println(str.substring(0, str.length() - 1));
      } // for (int trainingSetLoopIndex = 0; trainingSetLoopIndex < numTrainingSets; trainingSetLoopIndex++)
   } // public static void main(String[] args) throws IOException, FileNotFoundException
} // public class NetworkDebug2