package com.qczy.distillation.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.distillation.model.entity.MdLoraPresetEntity;
import org.apache.ibatis.annotations.*;

import java.util.List;

/**
 * LoRA配置预设 Mapper 接口
 *
 * 功能：
 * 1. LoRA预设的增删改查
 * 2. 按名称查询预设
 * 3. 查询所有可用预设
 *
 * @author AI Assistant
 * @date 2025-01-25
 */
@Mapper
public interface MdLoraPresetMapper extends BaseMapper<MdLoraPresetEntity> {

    /**
     * 根据预设名称查询预设
     * @param presetName 预设名称
     * @return LoRA预设实体
     */
    @Select("SELECT * FROM md_lora_preset WHERE preset_name = #{presetName} AND del_flag = 0")
    MdLoraPresetEntity selectByPresetName(@Param("presetName") String presetName);

    /**
     * 查询所有可用的LoRA预设
     * @return LoRA预设列表
     */
    @Select("SELECT * FROM md_lora_preset WHERE del_flag = 0 ORDER BY create_time DESC")
    List<MdLoraPresetEntity> selectAllPresets();

    /**
     * 逻辑删除预设
     * @param presetName 预设名称
     * @return 更新行数
     */
    @Update("UPDATE md_lora_preset SET del_flag = 1, update_time = NOW() " +
            "WHERE preset_name = #{presetName}")
    int deleteByPresetName(@Param("presetName") String presetName);

    /**
     * 检查预设名称是否已存在
     * @param presetName 预设名称
     * @return 数量
     */
    @Select("SELECT COUNT(*) FROM md_lora_preset WHERE preset_name = #{presetName} AND del_flag = 0")
    int countByPresetName(@Param("presetName") String presetName);
}
