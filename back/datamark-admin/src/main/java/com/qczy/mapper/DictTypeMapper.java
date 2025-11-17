package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.DictTypeEntity;
import com.qczy.model.entity.UserEntity;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/2 14:58
 * @Description:
 */
public interface DictTypeMapper extends BaseMapper<DictTypeEntity> {


    /**
     * 查询字典类型
     *
     * @param id 字典类型主键
     * @return 字典类型
     */
    public DictTypeEntity selectDictTypeById(Integer id);

    /**
     * 查询字典类型列表
     *
     * @param dictType 字典类型
     * @return 字典类型集合
     */
    IPage<DictTypeEntity>  selectDictTypeList(Page<DictTypeEntity> pageParam, @Param("req") DictTypeEntity dictType);

    /**
     * 新增字典类型
     *
     * @param dictType 字典类型
     * @return 结果
     */
    public int insertDictType(DictTypeEntity dictType);

    /**
     * 修改字典类型
     *
     * @param dictType 字典类型
     * @return 结果
     */
    public int updateDictType(DictTypeEntity dictType);

    /**
     * 删除字典类型
     *
     * @param id 字典类型主键
     * @return 结果
     */
    public int deleteDictTypeById(Integer id);

    /**
     * 批量删除字典类型
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteDictTypeByIds(int [] ids);
}
