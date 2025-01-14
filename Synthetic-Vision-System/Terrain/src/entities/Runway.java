package entities;


import java.util.ArrayList;
import java.util.List;

import org.lwjgl.util.vector.Vector2f;
import org.lwjgl.util.vector.Vector3f;

import data.normPoint;

public class Runway<Vertex2f>{
	
	private float width;
	private float length;
	private Vector2f anchor1;
	private Vector2f anchor2;
	private float elevation;
	private int runwayNumber;
	private Vector2f midpoint;
	private float heading;
	private float[] vertices;
	private int[] indices;
	private Vector2f p3;
	private Vector2f p4;
	
	
	public static Vector2f normalize(Vector2f init) {
		
		float x = init.getX();
		float y = init.getY();
		
		float squaredsum = (x * x) + (y * y);
		float magnitude = (float) Math.sqrt(squaredsum);
		
		return new Vector2f(x / magnitude, y / magnitude);
		
	}
	
	public Runway(float width, float length, Vector2f anchor1, Vector2f anchor2, float elevation, int runwayNumber) {
		
		this.width = width;
		this.length = length;
		this.anchor1 = anchor1;
		this.anchor2 = anchor2;
		this.elevation = elevation;
		this.runwayNumber = runwayNumber;
		this.midpoint = new Vector2f((this.anchor1.x + this.anchor2.x) / 2, (this.anchor1.y + this.anchor2.y) / 2);
		this.heading = calculateHeading();
		this.vertices = generateVertices();
		this.indices = generateIndices();
		
		
	}
	
	public Vector3f getMidpoint() {
		return new Vector3f(this.midpoint.x, this.midpoint.y, this.elevation);
	}
	
	
	public float[] generateVertices() {
		
		
		//VECTOR CALCULATIONS:
		
		Vector2f dirVec = new Vector2f(this.anchor1.getX() - this.anchor2.getX(), this.anchor1.getY() - this.anchor2.getY());
		Vector2f dot = new Vector2f(1, -1 * (dirVec.getY() / dirVec.getX()));
		
		if (this.isSameDirection()) {
			dot = normalize(dot);

		}
		else {
			dot = normalize(dot);
			dot.x = dot.getX() * -1;
			dot.y = dot.getY() * -1;
		}
		
		Vector2f pos1 = new Vector2f(this.anchor1.getX() + (dot.getX() * this.length), this.anchor1.getY() + (dot.getY() * this.length));
		Vector2f pos2 = new Vector2f(this.anchor2.getX() + (dot.getX() * this.length), this.anchor2.getY() + (dot.getY() * this.length));
		
	//	System.out.println(pos1);
	//	System.out.println(pos2);
		
		
		//BASE RUNWAY LEN 12:
		float[] base = {
				
			this.anchor1.getX(), this.elevation, this.anchor1.getY(),
			this.anchor2.getX(), this.elevation, this.anchor2.getY(),
			pos1.getX(), this.elevation, pos1.getY(),
			pos2.getX(), this.elevation, pos2.getY()
				
		};

		return base;
		
	}
	
	
	
	public int[] generateIndices() {
		
		//BASE RUNWAY:
		
		int[] ind = {
		2, 0, 1,
		2, 1, 3
		};
		
		return ind;
	}
	
	public float calculateHeading() {
		
		Vector2f dirVec = new Vector2f(this.anchor1.getX() - this.anchor2.getX(), this.anchor1.getY() - this.anchor2.getY());
		Vector2f dot = new Vector2f(1, -1 * (dirVec.getY() / dirVec.getX()));

		Vector2f norm = new Vector2f(0, 1);
		return Vector2f.angle(dot, norm);
	}
	
	public Vector3f centerlinePointDownDistance(float distance, boolean pos) {
		Vector2f dirVec = new Vector2f(this.anchor1.getX() - this.anchor2.getX(), this.anchor1.getY() - this.anchor2.getY());
		Vector2f dot = new Vector2f(-1 * dirVec.getY(), dirVec.getX());
		if (pos) {
			dot = normalize(dot);
		}
		else {
			dot.x = -1 * dot.getX();
			dot.y = -1 * dot.getY();
			dot = normalize(dot);
		}
		//System.out.println(dot);
		//System.out.println(dirVec);
		//System.out.println(Vector2f.dot(dot, dirVec));

		return new Vector3f(this.midpoint.getX() + (dot.getX() * distance), this.midpoint.getY() + (dot.getY() * distance), this.elevation);
		
		
	}
	
	public List<Vector3f> centerlinePositionGeneration(int markings){
		
		float distance = this.length / markings;
		List<Vector3f> centerlinePositions = new ArrayList<Vector3f>();
		
		for (int i = 0; i < markings; i++) {
			
			centerlinePositions.add(this.centerlinePointDownDistance(i * distance, false));
			
			
		}
		return centerlinePositions;
	}
	

	public boolean isSameDirection() {
		
		float adjustedHeading = (float) (Math.toDegrees(this.calculateHeading()));
		if (Math.abs(adjustedHeading - (10 * this.runwayNumber)) < 20) {
			return true;
		}
		else {
			return false;
		}
	}
	
	public List<Vector3f> singleCenterMarkingVertices(Vector3f centroid, float width, float length){
		
		Vector2f dirVec = new Vector2f(this.anchor1.getX() - this.anchor2.getX(), this.anchor1.getY() - this.anchor2.getY());
		Vector2f dot = new Vector2f(1, -1 * (dirVec.getY() / dirVec.getX()));
		
		dirVec = normalize(dirVec);
		dot = normalize(dot);
		
		Vector2f upperMiddle = new Vector2f(centroid.getX() * (dot.getX() * (length / 2)), centroid.getZ() * (dot.getY() * (length / 2)));
		Vector2f lowerMiddle = new Vector2f(centroid.getX() * (dot.getX() * -1 * (length / 2)), centroid.getZ() * (dot.getY() * -1 * (length / 2)));
		
		return null;

	}
	
	public boolean pointInRunway(Vector2f point) {
		
		return false;
	}
	
	
	public static void main(String[] args) {
		
		Runway r = new Runway(45, 1000, new Vector2f(500, 0), new Vector2f(0, 5000), 15, 5);
		System.out.println(r.centerlinePointDownDistance(1500, true));
	}
	
}