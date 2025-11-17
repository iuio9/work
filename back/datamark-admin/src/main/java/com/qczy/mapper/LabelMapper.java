package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.LabelEntity;
import com.qczy.model.entity.LabelGroupEntity;
import com.qczy.model.response.DataSetLabelResponse;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/19 15:17
 * @Description:
 */
public interface LabelMapper extends BaseMapper<LabelEntity> {


    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public LabelEntity selectLabelById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param label 【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    public IPage<LabelEntity> selectLabelList(Page<LabelEntity> pageParam, @Param("labelGroupId") Integer labelGroupId, @Param("req") LabelEntity label);

    /**
     * 新增【请填写功能名称】
     *
     * @param label 【请填写功能名称】
     * @return 结果
     */
    public int insertLabel(LabelEntity label);

    /**
     * 修改【请填写功能名称】
     *
     * @param label 【请填写功能名称】
     * @return 结果
     */
    public int updateLabel(LabelEntity label);

    /**
     * 删除【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteLabelById(Integer id);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteLabelByIds(int [] ids);

    List<DataSetLabelResponse> selectDataSetLabel(String sonId);

    List<LabelEntity> selectByIds(List<Integer> labelIds);

    boolean updateBatchById(List<LabelEntity> entityList);

    List<LabelEntity> selectListAll(List<Integer> labelId);
}
