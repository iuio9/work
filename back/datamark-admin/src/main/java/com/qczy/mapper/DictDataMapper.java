package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.DictDataEntity;
import com.qczy.model.entity.DictTypeEntity;
import com.qczy.model.response.DictSetTypeResponse;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/2 16:52
 * @Description:
 */
public interface DictDataMapper extends BaseMapper<DictDataEntity> {


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
     * @param pageParam 分页信息
     * @param typeId    类型id
     * @param dictData  查询类型
     * @return 【请填写功能名称】
     */
    IPage<DictDataEntity> selectDictDataList(Page<DictDataEntity> pageParam, @Param("typeId") Integer typeId, @Param("req") DictDataEntity dictData);

    /**
     * 新增【请填写功能名称】
     *
     * @param dictDataEntity 【请填写功能名称】
     * @return 结果
     */
    public int insertDictData(DictDataEntity dictDataEntity);

    /**
     * 修改【请填写功能名称】
     *
     * @param dictDataEntity 【请填写功能名称】
     * @return 结果
     */
    public int updateDictData(DictDataEntity dictDataEntity);

    /**
     * 删除【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteDictDataById(Integer id);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteDictDataByIds(int[] ids);

    List<DictSetTypeResponse> selectDataSetDictList();

    List<DictDataEntity> selectDictDataTreeSon(Integer dataTypeId);
}
