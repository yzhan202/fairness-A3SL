package binghamton.util;

import java.io.*; 
import java.util.*; 
import java.util.LinkedList; 


public class Graph {
	private int V;   // No. of vertices 
    private LinkedList<Integer> adj[]; //Adjacency List 
    private int startNode = 0;
    
    //Constructor 
    public Graph(int v) { 
        V = v; 
        adj = new LinkedList[v]; 
        for (int i=0; i<v; ++i) 
            adj[i] = new LinkedList(); 
    } 

    //Function to add an edge into the graph 
    public void addEdge(int v,int w) {  
    	adj[v].add(w);
    	startNode = v;
    } 
  
    // A recursive function to print DFS starting from v 
    public void DFSUtil(int v,Boolean visited[]) 
    { 
        // Mark the current node as visited and print it 
        visited[v] = true; 
  
        int n; 
  
        // Recur for all the vertices adjacent to this vertex 
        Iterator<Integer> i = adj[v].iterator(); 
        while (i.hasNext()) 
        { 
            n = i.next(); 
            if (!visited[n]) 
                DFSUtil(n,visited); 
        } 
    } 
  
  
    // The main function that returns true if graph is connected 
    public Boolean[] isCG() 
    { 
        // Step 1: Mark all the vertices as not visited 
        // (For first DFS) 
        Boolean visited[] = new Boolean[V]; 
        for (int i = 0; i < V; i++) 
            visited[i] = false; 
  
        // Step 2: Do DFS traversal starting from first vertex. 
        DFSUtil(startNode, visited); 
  
        // If DFS traversal doesn't visit all vertices, then 
        // return false. 
//        for (int i = 0; i < V; i++) 
//            if (visited[i] == false) 
//                return false; 
//        return true; 
        return visited;
    } 
}


