package binghamton.rl_fairness.A3C;


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


public class fairnessPSLModelCreation {
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
	 * Right Reason
	 */
	Set<String> Positive_Signal;
	Set<String> Negative_Signal;
	/*
	 * Fairness
	 */
	Set<StandardPredicate> SensitiveAttribute;
//	Set<String> highRankInstitutes;
//	Map<String, String> instituteMap;
	Set<String> protectedGroup;
	Map<String, String> paperAuthorMap;
	Map<String, Integer> truthMap;
	int protectedNum;
	int unprotectedNum;
	
	int[] protectedNegPos;
	int[] unprotectedNegPos;
	
	public fairnessPSLModelCreation(int threadNum) {
		dataDir = 'data/biasedPaperReview/bias4';
		// config manager
		cm = ConfigManager.getManager();
		config = cm.getBundle("SL_review"+ threadNum);
		Logger log = LoggerFactory.getLogger(this.class);
		
		// database
		def defaultPath = System.getProperty("java.io.tmpdir");
		String dbpath = config.getString("dbpath", defaultPath+ File.separator+ "a3cSL_review"+ threadNum);
		data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config);
		
		model = new PSLModel(this, data);
		
		// Default
		model.add predicate: "author", types: [ArgumentType.UniqueID]
		model.add predicate: "paper", types: [ArgumentType.UniqueID]
		model.add predicate: "reviewer", types: [ArgumentType.UniqueID]
		// Feature
		model.add predicate: "student", types: [ArgumentType.UniqueID]
		model.add predicate: "acceptable", types: [ArgumentType.UniqueID]
		model.add predicate: "reviews", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "submits", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		// Target
		model.add predicate: "positiveReviews", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "positiveSummary", types: [ArgumentType.UniqueID]
		// sensitive attribute
		model.add predicate: "institute", types: [ArgumentType.UniqueID]	
		model.add predicate: "highRank", types: [ArgumentType.UniqueID]
		model.add predicate: "affiliation", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		
		def closedPredicates = [author, paper, reviewer, student, acceptable, reviews, submits,
			institute, highRank, affiliation] as Set; 
		def inferredPredicates = [positiveReviews, positiveSummary] as Set; // presents
		
		negFeatPreds = [student, acceptable];
		
		// Right Reason
		Positive_Signal = ["ACCEPTABLE", "~STUDENT", "HIGHRANK", "POSITIVEREVIEWS", "POSITIVESUMMARY"] as Set;
		Negative_Signal = ["~ACCEPTABLE", "STUDENT", "~HIGHRANK", "~POSITIVEREVIEWS", "~POSITIVESUMMARY"] as Set;
		
		analysisSensitiveAttributes();
		
		def predicateFileMap = [
			((Predicate)author):"author.txt",
			((Predicate)paper):"paper.txt",
			((Predicate)reviewer):"reviewer.txt",
			((Predicate)student):"student.txt",
			((Predicate)acceptable):"acceptable.txt",
			((Predicate)reviews):"reviews.txt",
			((Predicate)submits):"submits.txt",
			((Predicate)positiveReviews):"positiveReviews.txt",
			((Predicate)positiveSummary):"positiveSummary.txt",
			
			((Predicate)institute):"institute.txt",
			((Predicate)highRank):"highRank.txt",
			((Predicate)affiliation):"affiliation.txt"]
		
		def predicateSoftTruthMap = [
			((Predicate)student):true,
			((Predicate)acceptable):true,
			((Predicate)author):false,
			((Predicate)paper):false,
			((Predicate)reviewer):false,
			((Predicate)reviews):false,
			((Predicate)submits):false,
			((Predicate)positiveReviews):true,
			((Predicate)positiveSummary):true,
			
			((Predicate)institute):false,
			((Predicate)highRank):true,
			((Predicate)affiliation):false]
		
		GenericVariable A = new GenericVariable('A', model); //author
		GenericVariable P = new GenericVariable("P", model);
		GenericVariable R = new GenericVariable("R", model);
		GenericVariable R2 = new GenericVariable("R2", model);
		
		GenericVariable I = new GenericVariable("I", model);
		
		generalPredArgsMap = [
			((Predicate)author): [[A]],
			((Predicate)paper): [[P]],
			((Predicate)reviewer): [[R], [R2]],
			((Predicate)student): [[A]],
			((Predicate)acceptable): [[P]],
			((Predicate)reviews): [[R,P], [R2,P]],
			((Predicate)submits): [[A,P]],
			((Predicate)positiveReviews): [[R,P], [R2,P]],
			((Predicate)positiveSummary): [[P]],
			
			((Predicate)institute): [[I]],
			((Predicate)highRank): [[I]],
			((Predicate)affiliation): [[A,I]]
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

	void populateDatabase(DataStore data, Database dbToPopulate, Partition populatePartition, Set inferredPredicates){
		Database populationDatabase = data.getDatabase(populatePartition, inferredPredicates);
		DatabasePopulator dbPop = new DatabasePopulator(dbToPopulate);
	
		for (Predicate p : inferredPredicates){
			dbPop.populateFromDB(populationDatabase, p);
		}
		populationDatabase.close();
	}
	
	void analysisSensitiveAttributes() {
		SensitiveAttribute = [institute, affiliation, highRank] as Set;
		
		Set<String> highRankInstitutes = new HashSet<>();
		Map<String, String> instituteMap = new HashMap<>();
		Set<String> studentSet = new HashSet<>();
		paperAuthorMap = new HashMap<>();
		protectedGroup = new HashSet<>();
		truthMap = new HashMap<>();
		
		//highRank
		String fileName = dataDir+ '/highRank.txt';
		BufferedReader reader = null;
		reader = new BufferedReader(new FileReader(fileName));
		String read = null;
		while((read = reader.readLine()) != null) {
			String[] splited = read.split(",");
			int value = Integer.parseInt(splited[1]);
			if (value == 1) {
				highRankInstitutes.add(splited[0]);
			}
		}
		// affiliation
		fileName = dataDir+ '/affiliation.txt';
		reader = new BufferedReader(new FileReader(fileName));
		read = null;
		while((read = reader.readLine()) != null) {
			String[] splited = read.split(",");
			String s = splited[0];
			String i = splited[1];
			instituteMap.put(s, i);
		}
		// student
		fileName = dataDir+ '/student.txt';
		reader = new BufferedReader(new FileReader(fileName));
		read = null;
		while((read = reader.readLine()) != null) {
			String[] splited = read.split(',');
			String s = splited[0];
			int value = Integer.parseInt(splited[1]);
			if (value == 1)
				studentSet.add(s);
		}
		// submits
		fileName = dataDir+ '/submits.txt';
		reader = new BufferedReader(new FileReader(fileName));
		read = null;
		while((read = reader.readLine()) != null) {
			String[] splited = read.split(',');
			String a = splited[0];
			String p = splited[1];
			paperAuthorMap.put(p, a);
		} 
		
		fileName = dataDir+ '/positiveSummary.txt'
		reader = new BufferedReader(new FileReader(fileName));
		read = null;
		while((read = reader.readLine()) != null) {
			String[] splited = read.split(",");
			int value = Integer.parseInt(splited[1]);
			truthMap.put(splited[0], value);
		}
		
		protectedNum = 0;
		unprotectedNum = 0;
		
		protectedNegPos = new int[2];
		unprotectedNegPos = new int[2];
		
        for (Map.Entry<String, String> entry : instituteMap.entrySet()) {
			String stud = entry.getKey();
			String inst = entry.getValue();
			if (studentSet.contains(stud)) {
				if (highRankInstitutes.contains(inst)) {
					unprotectedNum++; // Unprotected Group
				}
				else {
					protectedNum++; // Protected Group
					protectedGroup.add(stud);
				}
			} else {
				unprotectedNum++;
			}
		}
		
		for (Map.Entry<String, Integer> entry : truthMap.entrySet()) {
			String paper = entry.getKey();
			int value = entry.getValue();
			String author = paperAuthorMap.get(paper);
			if (protectedGroup.contains(author)) { // protected Group
				protectedNegPos[value] += 1;
			} else { // unprotected group
				unprotectedNegPos[value] += 1;
			}
		} 

	}
	
}










