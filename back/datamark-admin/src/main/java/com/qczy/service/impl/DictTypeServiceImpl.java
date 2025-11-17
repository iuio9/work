package com.qczy.service.impl;

import cn.hutool.core.convert.Convert;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import com.qczy.common.constant.SystemConstant;
import com.qczy.mapper.DictTypeMapper;
import com.qczy.model.entity.DictTypeEntity;
import com.qczy.model.entity.UserEntity;
import com.qczy.service.DictTypeService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.*;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/2 15:33
 * @Description:
 */
@Service
public class DictTypeServiceImpl extends ServiceImpl<DictTypeMapper, DictTypeEntity> implements DictTypeService {


    @Autowired
    private DictTypeMapper dictTypeMapper;

    /**
     * 查询字典类型
     *
     * @param id 字典类型主键
     * @return 字典类型
     */
    @Override
    public DictTypeEntity selectDictTypeById(Integer id) {
        return dictTypeMapper.selectDictTypeById(id);
    }

    /**
     * 查询字典类型列表
     *
     * @param dictType 字典类型
     * @return 字典类型
     */
    @Override
    public IPage<DictTypeEntity> selectDictTypeList(Page<DictTypeEntity> pageParam, DictTypeEntity dictType) {
        return dictTypeMapper.selectDictTypeList(pageParam,dictType);
    }

    /**
     * 新增字典类型
     *
     * @param dictType 字典类型
     * @return 结果
     */
    @Override
    public int insertDictType(DictTypeEntity dictType) {
        dictType.setCreateTime(new Date());
        dictType.setIsAllowDeletion(SystemConstant.YES_DISABLE_DATA);
        return dictTypeMapper.insertDictType(dictType);
    }

    /**
     * 修改字典类型
     *
     * @param dictType 字典类型
     * @return 结果
     */
    @Override
    public int updateDictType(DictTypeEntity dictType) {
        dictType.setUpdateTime(new Date());
        dictType.setIsAllowDeletion(SystemConstant.YES_DISABLE_DATA);
        return dictTypeMapper.updateDictType(dictType);
    }

    /**
     * 批量删除字典类型
     *
     * @param ids 需要删除的字典类型主键
     * @return 结果
     */
    @Override
    public int deleteDictTypeByIds(String ids) {
        return 1;
        //return dictTypeMapper.deleteDictTypeByIds();

    }

    /**
     * 删除字典类型信息
     *
     * @param id 字典类型主键
     * @return 结果
     */
    @Override
    public int deleteDictTypeById(Integer id) {
        return dictTypeMapper.deleteDictTypeById(id);
    }

    @Override
    public int MyDeleteAll(int [] ids) {

        return dictTypeMapper.deleteDictTypeByIds(ids);
    }

    @Override
    public List<DictTypeEntity> selectDictType() {
        return dictTypeMapper.selectList(null);
    }

}
