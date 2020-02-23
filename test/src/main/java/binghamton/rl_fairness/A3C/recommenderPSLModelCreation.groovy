package binghamton.rl_fairness.A3C

import java.text.DecimalFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import edu.umd.cs.psl.application.inference.*;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.em.HardEM
import edu.umd.cs.psl.config.*;
import edu.umd.cs.psl.core.*;
import edu.umd.cs.psl.core.inference.*;
import edu.umd.cs.psl.database.*;
import edu.umd.cs.psl.database.rdbms.*;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver;
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type;
import edu.umd.cs.psl.evaluation.result.*;
import edu.umd.cs.psl.evaluation.statistics.*;
import edu.umd.cs.psl.groovy.*;
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.syntax.FormulaContainer
import edu.umd.cs.psl.groovy.syntax.GenericVariable
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.atom.*;
import edu.umd.cs.psl.model.formula.*;
import edu.umd.cs.psl.model.function.*;
import edu.umd.cs.psl.model.kernel.*;
import edu.umd.cs.psl.model.kernel.rule.AbstractRuleKernel
import edu.umd.cs.psl.model.kernel.rule.CompatibilityRuleKernel
import edu.umd.cs.psl.model.predicate.*;
import edu.umd.cs.psl.model.term.*;
import edu.umd.cs.psl.model.rule.*;
import edu.umd.cs.psl.model.weight.*;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.Term
import edu.umd.cs.psl.model.argument.UniqueID
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.ui.loading.*;
import edu.umd.cs.psl.util.database.*;
import com.google.common.collect.Iterables;
import edu.umd.cs.psl.util.database.Queries;
import edu.umd.cs.psl.evaluation.resultui.printer.*;
import java.io.*;
import java.util.*;
import groovy.time.*;

import binghamton.util.FoldUtils
import binghamton.util.GroundingWrapper
import binghamton.util.DataOutputter;


public class recommenderPSLModelCreation {
	String dataDir;
	ConfigManager cm;
	ConfigBundle config;
	DataStore data;
	PSLModel model;
	
	Map<StandardPredicate, List<List<GenericVariable>>> generalPredArgsMap;
	
	Database wlTruthDB;
	Partition trainPart;
	
	StandardPredicate[] X;
	StandardPredicate[] Y;
	StandardPredicate[] Z;
	StandardPredicate[] negFeatPreds;
	
	/*
	 * Right Reasons
	 */
	Set<String> Positive_Signal;
	Set<String> Negative_Signal;
	/*
	 * Fairness
	 */
	Set<StandardPredicate> SensitiveAttribute;
	Set<Integer> protectedGroup;
	Map<String, Double> truthMap;
	Set<Integer> ItemSet;
	Map<Integer, Integer> protectedItemNum;
	Map<Integer, Integer> unprotectedItemNum;
	int protectedNum;
	int unprotectedNum;
	Map<Integer, Double> aveProtectedItemScore;
	Map<Integer, Double> aveUnprotectedItemScore;
	
	public recommenderPSLModelCreation(int threadNum) {
		dataDir = 'data/movie_cv5_100k/'+ 0+'/training';
		// config manager
		cm = ConfigManager.getManager();
		config = cm.getBundle("SL_movielens"+threadNum);
		Logger log = LoggerFactory.getLogger(this.class);
		
		// database
		def defaultPath = System.getProperty("java.io.tmpdir");
		String dbpath = config.getString("dbpath", defaultPath+ File.separator+ "a3cSL_movielens"+ threadNum);
		data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config);
		
		model = new PSLModel(this, data);
		
		// Default
		model.add predicate: "user", types: [ArgumentType.UniqueID]
		model.add predicate: "item", types: [ArgumentType.UniqueID]
		model.add predicate: "reviews", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "userLink", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "itemLink", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		// Feature
//		model.add predicate: "metaItemSim", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "metaUserSim", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "itemPearsonSim", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "userPearsonSim", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		
		model.add predicate: "avgUserRating", types: [ArgumentType.UniqueID]
		model.add predicate: "avgItemRating", types: [ArgumentType.UniqueID]
		model.add predicate: "cf", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "bpmf", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		// Target
		model.add predicate: "rating", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		// Sensitive Attribute
		model.add predicate: "gender", types: [ArgumentType.UniqueID]
		
		def closedPredicates = [user, item, reviews, userLink, itemLink, metaUserSim, itemPearsonSim, userPearsonSim, 
			avgUserRating, avgItemRating, cf, bpmf, gender] as Set; // metaItemSim, , 
		def inferredPredicates = [rating] as Set;
		
		negFeatPreds = [avgUserRating, avgItemRating, cf, bpmf];
		
		// Right Reason
		Positive_Signal = ["AVGUSERRATING", "AVGITEMRATING", "CF", "RATING"] as Set;
		Negative_Signal = ["~AVGUSERRATING", "~AVGITEMRATING", "~CF", "~RATING"] as Set;
		
		analysisSensitiveAttributes();
		
		def predicateFileMap = [
			((Predicate)user):"user.txt",
			((Predicate)item):"item.txt",
			((Predicate)reviews):"review.txt", 
			((Predicate)userLink):"userLink.txt",
			((Predicate)itemLink):"itemLink.txt", 
//			((Predicate)metaItemSim):"metaItemSim.txt",
			((Predicate)metaUserSim):"metaUserSim.txt",
			((Predicate)itemPearsonSim):"itemPearsonSim.txt",
			((Predicate)userPearsonSim):"userPearsonSim.txt",
			
			((Predicate)avgUserRating):"avgUserRating.txt",
			((Predicate)avgItemRating):"avgItemRating.txt",
			((Predicate)cf):"cf.txt", // "cf.txt"; "fairCF.txt"
			((Predicate)bpmf): "bpmf.txt", 
			
			((Predicate)rating):"rating.txt",
			((Predicate)gender):"gender.txt"]

		def predicateSoftTruthMap = [
			((Predicate)user):false,
			((Predicate)item):false,
			((Predicate)reviews):false, 
			((Predicate)userLink):false,
			((Predicate)itemLink):false,
//			((Predicate)metaItemSim):false,
			((Predicate)metaUserSim):false,
			((Predicate)itemPearsonSim):true,
			((Predicate)userPearsonSim):true,
			
			((Predicate)avgUserRating):true,
			((Predicate)avgItemRating):true,
			((Predicate)cf):true,
			((Predicate)bpmf): true, 
			
			((Predicate)rating):true,
			((Predicate)gender):true]
		
		GenericVariable U = new GenericVariable('U', model);
		GenericVariable U2 = new GenericVariable('U2', model);
		GenericVariable I = new GenericVariable('I', model);
		GenericVariable I2 = new GenericVariable('I2', model);
		
		generalPredArgsMap = [
			((Predicate)user): [[U], [U2]],
			((Predicate)item): [[I], [I2]],
			((Predicate)reviews): [[U,I], [U,I2], [U2,I]],
			((Predicate)userLink): [[U,U2]],
			((Predicate)itemLink): [[I, I2]],  
//			((Predicate)metaItemSim): [[I,I2]],
			((Predicate)metaUserSim): [[U,U2]],
			((Predicate)itemPearsonSim): [[I,I2]],
			((Predicate)userPearsonSim): [[U,U2]],
			
			((Predicate)avgUserRating): [[U], [U2]],
			((Predicate)avgItemRating): [[I], [I2]],
			((Predicate)cf): [[U,I]], // , [U,I2], [U2,I]
			((Predicate)bpmf): [[U,I]], 
			((Predicate)rating): [[U,I], [U,I2], [U2,I]]
			]
		
		trainPart = new Partition(threadNum*1000+ 0);
		Partition truthPart = new Partition(threadNum*1000+ 1);
		
		def inserter;
		for (Predicate p: closedPredicates) {
			String fileName = predicateFileMap[p];
			inserter = data.getInserter(p, trainPart);
			def fullFilePath = dataDir+ '/'+ fileName;
//			println p.toString()
			if (predicateSoftTruthMap[p]) {
				InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
			} else {
				InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
			}
		}
		
		for (Predicate p: inferredPredicates) {
			String fileName = predicateFileMap[p];
			inserter = data.getInserter(p, truthPart);
			def fullFilePath = dataDir + '/' + fileName;
			if(predicateSoftTruthMap[p]){
				InserterUtils.loadDelimitedDataTruth(inserter, fullFilePath, ',');
			}
			else{
				InserterUtils.loadDelimitedData(inserter, fullFilePath, ',');
			}
		}
		
		wlTruthDB = data.getDatabase(truthPart, inferredPredicates);
		
		X = closedPredicates.toArray();
		Y = inferredPredicates.toArray();
		Z = [];
 	}
		
	void analysisSensitiveAttributes() {
		sensitiveAttribute = [gender] as Set;
		protectedGroup = new HashSet<>();
		String fileName = dataDir+'/gender.txt';
		BufferedReader reader = null;
		reader = new BufferedReader(new FileReader(fileName));
		String read = null;
		while((read=reader.readLine()) != null) {
			String[] splited = read.split(",");
			int u = Integer.parseInt(splited[0]);
			int g = Integer.parseInt(splited[1]);
			if (g == 0) { // Protected or Disadvantaged Group
				protectedGroup.add(u);
			}
		}
		
		truthMap = new HashMap<>();
		ItemSet = new HashSet<>();
		protectedItemNum = new HashMap<>();
		unprotectedItemNum = new HashMap<>();
		protectedNum = 0;
		unprotectedNum = 0;
		aveProtectedItemScore = new HashMap<>();
		aveUnprotectedItemScore = new HashMap<>();
		fileName = dataDir+'/rating.txt';
		reader = new BufferedReader(new FileReader(fileName));
		read = null;
		while((read=reader.readLine()) != null) {
			String[] splited = read.split(",");
			String key = splited[0]+','+splited[1];
			double score = Double.valueOf(splited[2]);
			truthMap.put(key, score);
			
			int u = Integer.parseInt(splited[0]);
			int i = Integer.parseInt(splited[1]);
			if (protectedGroup.contains(u)) { // Protected Group
				protectedNum++;
				if (protectedItemNum.containsKey(i)) {
					int num = protectedItemNum.get(i);
					protectedItemNum.put(i, num+1);
					
					double tmp = aveProtectedItemScore.get(i);
					aveProtectedItemScore.put(i, tmp+score);
				} else {
					protectedItemNum.put(i, 1);
					aveProtectedItemScore.put(i, score)
				}
				
			} else { // Unprotected Group
				unprotectedNum++;
				if (unprotectedItemNum.containsKey(i)) {
					int num = unprotectedItemNum.get(i);
					unprotectedItemNum.put(i, num+1);
					
					double tmp = aveUnprotectedItemScore.get(i);
					aveUnprotectedItemScore.put(i, tmp+score);
				} else {
					unprotectedItemNum.put(i, 1);
					aveUnprotectedItemScore.put(i, score);
				}
			}
		}
		
		fileName = dataDir+'/item.txt';
		reader = new BufferedReader(new FileReader(fileName));
		read = null;
		while((read=reader.readLine()) != null) {
			int v = Integer.parseInt(read);
			ItemSet.add(v);
		}
		
		for (int v : protectedItemNum.keySet()) {
			int num = protectedItemNum.get(v);
			double sum = aveProtectedItemScore.get(v);
			aveProtectedItemScore.put(v, sum/(1.0*num));
		}
		for (int v : unprotectedItemNum.keySet()) {
			int num = unprotectedItemNum.get(v);
			double sum = aveUnprotectedItemScore.get(v);
			aveUnprotectedItemScore.put(v, sum/(1.0*num));
		}
		
	}
	
}





