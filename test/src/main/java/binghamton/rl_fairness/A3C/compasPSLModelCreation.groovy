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



public class compasPSLModelCreation {
	String dataDir;
	ConfigManager cm;
	ConfigBundle config;
	DataStore data;
	PSLModel model;
	final int fold = 4;
	
	Map<StandardPredicate, List<GenericVariable>> generalPredArgsMap;
	Database wlTruthDB;
	Partition trainPart;
	
	StandardPredicate[] X;
	StandardPredicate[] Y;
	StandardPredicate[] Z;
	StandardPredicate[] negFeatPreds;
	
	StandardPredicate dummyPred;
	
	/*
	 * Right Reason
	 */
	Set<String> Recid_Signal;
	Set<String> Inrecid_Signal;
	
	// Fairness
	Set<StandardPredicate> SensitiveAttributes;
	Set<String> protectedGroup;
	int protectedNum;
	int unprotectedNum;
	
	public compasPSLModelCreation(int threadNum) {
		dataDir = 'data/compas_5cv/'+fold+ '/training'
		// config manager
		cm = ConfigManager.getManager();
		config = cm.getBundle("SL_review"+ threadNum);
		Logger log = LoggerFactory.getLogger(this.class);
		
		// Database
		def defaultPath = System.getProperty("java.io.tmpdir");
		String dbpath = config.getString("dbpath", defaultPath+ File.separator+ "a3cSL_review"+ threadNum);
		data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config);
		
		model = new PSLModel(this, data);
		
		model.add predicate: "user", types: [ArgumentType.UniqueID]
		model.add predicate: "recid", types: [ArgumentType.UniqueID]
		// Age
		model.add predicate: "oldAge", types: [ArgumentType.UniqueID]
		// Charge Degree
		model.add predicate: "felony", types: [ArgumentType.UniqueID]
		// Prison History
		model.add predicate: "juvFelHistory", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "juvMisdHistory", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "juvOtherHistory", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "priors", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "priorFelony", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "priorMisd", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		model.add predicate: "priorOther", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		
		// Days in Jail
		model.add predicate: "longJailDay", types: [ArgumentType.UniqueID]
		
		// Sensitive Attributes
		model.add predicate: "race", types: [ArgumentType.UniqueID, ArgumentType.String]
		model.add predicate: "sex", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		
		def closedPredicates = [user, oldAge, felony, juvFelHistory, juvMisdHistory, juvOtherHistory, 
			priors, priorFelony, priorMisd, priorOther, longJailDay, 
			race, sex] as Set; 
		def inferredPredicates = [recid] as Set;
		
		negFeatPreds = [oldAge, felony, longJailDay];
		dummyPred = user;
		
		Recid_Signal = ["~OLDAGE", "FELONY", "JUVFELHISTORY", "JUVMISDHISTORY", "JUVOTHERHISTORY",
			"PRIORS", "PRIORFELONY", "PRIORMISD", "PRIOROTHER", "LONGJAILDAY"] as Set;
		Inrecid_Signal = ["OLDAGE", "~FELONY", "~LONGJAILDAY"] as Set;
		
		analysisSensitiveAttributes();
		
		def predicateFileMap = [
			((Predicate)user): "user.txt",
			((Predicate)recid): "recidivist.txt",
			((Predicate)oldAge): "oldAge.txt",
			((Predicate)felony): "felony.txt",
			((Predicate)juvFelHistory): "juvFelHistory.txt",
			((Predicate)juvMisdHistory): "juvMisdHistory.txt",
			((Predicate)juvOtherHistory): "juvOtherHistory.txt",
			((Predicate)priors): "priors.txt",
			((Predicate)priorFelony): "priorChargeFelony.txt",
			((Predicate)priorMisd): "priorChargeMisd.txt",
			((Predicate)priorOther): "priorChargeOther.txt",
			((Predicate)longJailDay): "longJailDay.txt",
			((Predicate)race): "race.txt",
			((Predicate)sex): "sex.txt"]
	
		def predicateSoftTruthMap = [
			((Predicate)user): false,
			((Predicate)recid): true,
			((Predicate)oldAge): true,
			((Predicate)felony): true,
			((Predicate)juvFelHistory): false,
			((Predicate)juvMisdHistory): false,
			((Predicate)juvOtherHistory): false,
			((Predicate)priors): false,
			((Predicate)priorFelony): false,
			((Predicate)priorMisd): false,
			((Predicate)priorOther): false,
			((Predicate)longJailDay): true,
			((Predicate)race): false,
			((Predicate)sex): false]
		
		GenericVariable U = new GenericVariable("U", model);
		GenericVariable I1 = new GenericVariable("I1", model);
		GenericVariable I2 = new GenericVariable("I2", model);
		GenericVariable I3 = new GenericVariable("I3", model);
		GenericVariable I4 = new GenericVariable("I4", model);
		GenericVariable I5 = new GenericVariable("I5", model);
		GenericVariable I6 = new GenericVariable("I6", model);
		GenericVariable I7 = new GenericVariable("I7", model);
		
		GenericVariable A = new GenericVariable("A", model);
		
		generalPredArgsMap = [
			((Predicate)user): [U],
			((Predicate)recid): [U],
			((Predicate)oldAge): [U],
			((Predicate)felony): [U],
			((Predicate)juvFelHistory): [U, I1],
			((Predicate)juvMisdHistory): [U, I2],
			((Predicate)juvOtherHistory): [U, I3],
			((Predicate)priors): [U, I4],
			((Predicate)priorFelony): [U, I5],
			((Predicate)priorMisd): [U, I6],
			((Predicate)priorOther): [U, I7],
			((Predicate)longJailDay): [U],
			((Predicate)race): [U, A],
			((Predicate)sex): [U]]
		
		trainPart = new Partition(threadNum*500+0);
		Partition truthPart = new Partition(threadNum*500+1);
		
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
		protectedGroup = new HashSet<String>();
		protectedNum = 0;
		unprotectedNum = 0;
		
		SensitiveAttributes = [race, sex] as Set;
		
		String raceFile = dataDir+ '/race.txt';
		BufferedReader reader = null;
		reader = new BufferedReader(new FileReader(raceFile));
		String read = null;
		while((read = reader.readLine()) != null) {
			String[] splited = read.split(',')
			String user = splited[0];
			String race = splited[1];
			if (race.equals("African-American")) {
				protectedGroup.add(user);
				protectedNum++;
			} else {
				unprotectedNum++;
			}
		}
	}
	
}








