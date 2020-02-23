package binghamton.rl_fairness.A3C.compas_fairness;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.stream.IntStream;

import org.apache.commons.lang3.ArrayUtils;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.jcodec.common.ArrayUtil;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.omg.CORBA.Current;

import binghamton.rl.A3C.ActorCriticLoss;
import binghamton.rl.A3C.ActorCriticSeparate;
import edu.umd.cs.psl.config.ConfigBundle;
import edu.umd.cs.psl.database.DataStore;
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.GenericVariable;
import edu.umd.cs.psl.model.predicate.PredicateFactory;
import edu.umd.cs.psl.model.predicate.StandardPredicate;

import binghamton.rl_fairness.A3C.compasPSLModelCreation;


public class a3cMDP_compas {
	final int maxEpochStep;
	final int numThread;
	final int nstep; // t_max
	final double gamma;
	final double delta; 
	
	final int maxRuleLen;
	final int maxRuleNum;
	
	AsyncGlobal_compas asyncGlobal;
	
	public a3cMDP_compas() {
		this.maxRuleLen = 3; // 4
		this.maxRuleNum = 12; // 15; 10
		
		this.maxEpochStep = (int)1.0e+5;
		this.numThread = 2;
		this.nstep = 6;
		this.gamma = 1.0;
		this.delta = 0.1;
		
		compasPSLModelCreation pslGenerator = new compasPSLModelCreation(66);
		StandardPredicate[] X = pslGenerator.getX();
		StandardPredicate[] Y = pslGenerator.getY();
		StandardPredicate[] Z = pslGenerator.getZ();
		StandardPredicate dummyPred = pslGenerator.getDummyPred();
		Set<StandardPredicate> sensitiveAttrs = pslGenerator.getSensitiveAttributes();
		
		X = ArrayUtils.removeElement(X, dummyPred);
		for (StandardPredicate p : sensitiveAttrs) {
			X = ArrayUtils.removeElement(X, p);
		}
		StandardPredicate[] negFeatPreds = pslGenerator.getNegFeatPreds();
		
		StandardPredicate[] allPosPreds = ArrayUtils.addAll(ArrayUtils.addAll(X, Y), Z);
		StandardPredicate[] allNegPreds = ArrayUtils.addAll(ArrayUtils.addAll(negFeatPreds, Y), Z);
		int posPredNum = allPosPreds.length;
		int negPredNum = allNegPreds.length;
		
		final int outputSize = posPredNum+ negPredNum+ 1; // // Positive and Negative Predicates and "Return"
		final int inputRow = maxRuleNum;
		final int inputCol = outputSize;
		
		ActorCriticSeparate actorCriticNN = buildActorCritic(new int[] {inputRow, inputCol}, outputSize);
		asyncGlobal = new AsyncGlobal_compas(actorCriticNN, maxEpochStep);
	}
	
	public ActorCriticSeparate buildActorCritic(int[] numInputs, int numOutputs) {				
		final int NEURAL_NET_SEED = 12345;
		final double l2 = 0; //1e-5; //0.0001;
		final int numHiddenNodes = 18; //24; 20; 18;
		final int numLayers = 3;
		final boolean isUseLSTM = false; //true;
		final int NEURAL_NET_ITERATION_LISTENER = 5000;
		final double adam = 1e-3; // 0.001
		
		int nIn = 1;
		for (int i: numInputs) {
			nIn *= i;
		}
		System.out.println("nIn: "+ nIn);
		
		/*
		 * Value Net
		 */
		NeuralNetConfiguration.ListBuilder confB = new NeuralNetConfiguration.Builder().seed(NEURAL_NET_SEED)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(new Adam(adam))
				.weightInit(WeightInit.XAVIER)
				.l2(l2)
				.list().layer(0, new DenseLayer.Builder().nIn(nIn).nOut(numHiddenNodes)
					.activation(Activation.TANH).build()); //RELU
		for (int i=1; i<numLayers; i++) {
			confB.layer(i, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                    .activation(Activation.TANH).build());
		}	
		confB.layer(numLayers, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
                .nIn(numHiddenNodes).nOut(1).build());
		
		confB.setInputType(InputType.feedForward(nIn));
		MultiLayerConfiguration mlnconf = confB.pretrain(false).backprop(true).build();
        MultiLayerNetwork valueNet_model = new MultiLayerNetwork(mlnconf);
        valueNet_model.init();
        valueNet_model.setListeners(new ScoreIterationListener(NEURAL_NET_ITERATION_LISTENER));
		
        /*
         * Policy Net
         */
        NeuralNetConfiguration.ListBuilder confB2 = new NeuralNetConfiguration.Builder().seed(NEURAL_NET_SEED)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Adam(adam))
                .weightInit(WeightInit.XAVIER)
//                .regularization(true)
                .l2(l2)
                .list().layer(0, new DenseLayer.Builder().nIn(nIn).nOut(numHiddenNodes)
                                .activation(Activation.TANH).build()); //RELU
        for (int i = 1; i < numLayers; i++) {
            confB2.layer(i, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                            .activation(Activation.TANH).build());
        }
        confB2.layer(numLayers, new OutputLayer.Builder(new ActorCriticLoss())
                        .activation(Activation.SOFTMAX).nIn(numHiddenNodes).nOut(numOutputs).build());
        
        confB2.setInputType(InputType.feedForward(nIn));
        MultiLayerConfiguration mlnconf2 = confB2.pretrain(false).backprop(true).build();
        MultiLayerNetwork policyNet_model = new MultiLayerNetwork(mlnconf2);
        policyNet_model.init();
        policyNet_model.setListeners(new ScoreIterationListener(NEURAL_NET_ITERATION_LISTENER));
        
		return new ActorCriticSeparate(valueNet_model, policyNet_model);
	}
	
	public AsyncThread_compas newThread(int i) {
		return new AsyncThread_compas(asyncGlobal, i, maxRuleLen, maxRuleNum, nstep, gamma, delta);
	}
	
	protected boolean isTrainingComplete() {
		return asyncGlobal.isTrainingComplete();
	}
	
	public int getStepCounter() {
		return asyncGlobal.getT().get();
	}
	
	public void train() {
		try {
			System.out.println("AsyncLearning training Starting.");
			asyncGlobal.start();
			
			for (int i=0; i<numThread; i++) {
				Thread t = newThread(i);
				Nd4j.getAffinityManager().attachThreadToDevice(t, i % Nd4j.getAffinityManager().getNumberOfDevices());
				t.start();
				
//				AsyncThread_fairness t = newThread(i);
//				t.run();
			}
			
			asyncGlobal.join();
		} catch (Exception e) {
			System.out.println("Training Failed. "+ e);
			e.printStackTrace();
		}
	}
	
	public void saveA3C(String path1, String path2) throws IOException {
		asyncGlobal.current.save(path1, path2);
	}
	
	public void loadA3C(String path0, String path1) throws IOException {
		asyncGlobal.current = ActorCriticSeparate.load(path0, path1);
	}
}










