package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.DeptEntity;
import com.qczy.model.entity.DictDataEntity;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/6 14:43
 * @Description:
 */
public interface DeptMapper extends BaseMapper<DeptEntity> {

    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public DeptEntity selectDeptById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param Dept 【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    IPage<DeptEntity> selectDeptList(Page<DeptEntity> pageParam,@Param("req") DeptEntity Dept);

    /**
     * 新增【请填写功能名称】
     *
     * @param Dept 【请填写功能名称】
     * @return 结果
     */
    public int insertDept(DeptEntity Dept);

    /**
     * 修改【请填写功能名称】
     *
     * @param Dept 【请填写功能名称】
     * @return 结果
     */
    public int updateDept(DeptEntity Dept);

    /**
     * 删除【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteDeptById(Integer id);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteDeptByIds(int[] ids);

}
