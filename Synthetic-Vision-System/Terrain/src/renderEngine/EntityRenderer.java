package renderEngine;
 
import models.RawModel;
import models.TexturedModel;

import java.util.List;
import java.util.Map;

import org.lwjgl.input.Keyboard;
import org.lwjgl.opengl.Display;
import org.lwjgl.opengl.GL11;
import org.lwjgl.opengl.GL13;
import org.lwjgl.opengl.GL20;
import org.lwjgl.opengl.GL30;
import org.lwjgl.util.vector.Matrix4f;
 
import shaders.StaticShader;
import textures.ModelTexture;
import toolbox.Maths;
 
import entities.Entity;
 
public class EntityRenderer {
     

    private StaticShader shader;
    
    public EntityRenderer(StaticShader shader, Matrix4f projectionMatrix){
    	this.shader = shader;
    	
        shader.start();
        shader.loadProjectionMatrix(projectionMatrix);
        shader.stop();
    }

	public void render(Map<TexturedModel, List<Entity>> entities) {
    	for(TexturedModel model: entities.keySet()){
    		prepareTexturedModel(model);
    		List<Entity> batch = entities.get(model);
    		boolean t = false;
    		for(Entity entity: batch) {
    			prepareInstance(entity);
    			
    			if (Keyboard.isKeyDown(Keyboard.KEY_Y)) {
    				t = true;
    			}
    			
    			if (t){
        			GL11.glPolygonMode(GL11.GL_FRONT_AND_BACK, GL11.GL_LINE);
    				GL11.glDrawElements(GL11.GL_LINES, model.getRawModel().getVertexCount(), GL11.GL_UNSIGNED_INT, 0);


    			}
    			else {
    				GL11.glPolygonMode(GL11.GL_FRONT_AND_BACK, GL11.GL_FILL);
    				GL11.glDrawElements(GL11.GL_TRIANGLES, model.getRawModel().getVertexCount(), GL11.GL_UNSIGNED_INT, 0);
    			}
    		}
    		unbindTexturedModel();
    	}
    }
    
    public void renderWire(Map<TexturedModel, List<Entity>> entities) {
    	for(TexturedModel model: entities.keySet()){
    		prepareTexturedModel(model);
    		List<Entity> batch = entities.get(model);
    		for(Entity entity: batch) {
    			prepareInstance(entity);
    			
        		GL11.glPolygonMode(GL11.GL_FRONT_AND_BACK, GL11.GL_LINE);
    			GL11.glDrawElements(GL11.GL_LINES, model.getRawModel().getVertexCount(), GL11.GL_UNSIGNED_INT, 0);

    			
    		}
    		unbindTexturedModel();
    	}
    }
    
    private void prepareTexturedModel(TexturedModel model) {
        RawModel rawModel = model.getRawModel();
        GL30.glBindVertexArray(rawModel.getVaoID());
        GL20.glEnableVertexAttribArray(0);
        GL20.glEnableVertexAttribArray(1);
        GL20.glEnableVertexAttribArray(2);
        
        ModelTexture texture = model.getTexture();
        shader.loadShineVariables(texture.getShineDamper(), texture.getReflectivity());
        GL13.glActiveTexture(GL13.GL_TEXTURE0);
        GL11.glBindTexture(GL11.GL_TEXTURE_2D, model.getTexture().getID());
    }
    
    private void unbindTexturedModel() {
        GL20.glDisableVertexAttribArray(0);
        GL20.glDisableVertexAttribArray(1);
        GL20.glDisableVertexAttribArray(2);
        GL30.glBindVertexArray(0);
    }
    
    private void prepareInstance(Entity entity) {
    	Matrix4f transformationMatrix = Maths.createTransformationMatrix(entity.getPosition(),
                entity.getRotX(), entity.getRotY(), entity.getRotZ(), entity.getScale());
        shader.loadTransformationMatrix(transformationMatrix);
    }
   

 
}