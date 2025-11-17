package com.qczy.service;

import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.baomidou.mybatisplus.extension.service.IService;
import com.qczy.mapper.DictDataMapper;
import com.qczy.model.entity.DictDataEntity;
import com.qczy.model.entity.DictTypeEntity;
import com.qczy.model.response.DictDataTreeResponse;
import com.qczy.model.response.DictSetTypeResponse;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/5 9:52
 * @Description:
 */
public interface DictDataService extends IService<DictDataEntity> {


    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public DictDataEntity selectDictDataById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param dictData 【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    IPage<DictDataEntity> selectDictDataList(Page<DictDataEntity> pageParam, Integer typeId, DictDataEntity dictData);

    /**
     * 新增【请填写功能名称】
     *
     * @param dictData 【请填写功能名称】
     * @return 结果
     */
    public int insertDictData(DictDataEntity dictData);

    /**
     * 修改【请填写功能名称】
     *
     * @param dictData 【请填写功能名称】
     * @return 结果
     */
    public int updateDictData(DictDataEntity dictData);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的【请填写功能名称】主键集合
     * @return 结果
     */
    public int deleteDictDataByIds(int [] ids);

    /**
     * 删除【请填写功能名称】信息
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteDictDataById(Integer id);

    List<DictDataEntity> getDictDataTree(Integer typeId);


    List<DictSetTypeResponse> selectDataSetDictList();

    List<DictDataEntity> selectBatchId(Integer[] ids);

    String getTreeLevelDict(Integer dataTypeId);

    int getFatherAndDataSon(int[] ids);

    List<Integer> getTreeLevelDictIds(Integer dataTypeId);
}
