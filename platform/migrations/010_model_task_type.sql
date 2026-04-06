-- Add task_type and algorithm columns to model_registry for multi-task ML pipeline.
-- task_type: "locomotion", "reach", "grasp", "pick_place"
-- algorithm: "ppo", "sac", "bc" (behavior cloning)
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS task_type VARCHAR(64) NOT NULL DEFAULT 'locomotion';
ALTER TABLE model_registry ADD COLUMN IF NOT EXISTS algorithm VARCHAR(32) NOT NULL DEFAULT 'ppo';
CREATE INDEX IF NOT EXISTS idx_model_registry_task ON model_registry(task_type, status);
