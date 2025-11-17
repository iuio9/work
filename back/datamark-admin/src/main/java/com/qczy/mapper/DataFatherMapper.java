package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.baomidou.mybatisplus.core.metadata.IPage;
import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import com.qczy.model.entity.DataFatherEntity;
import com.qczy.model.request.DataSonQueryRequest;
import com.qczy.model.response.DataResponse;
import org.apache.ibatis.annotations.Param;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 18:44
 * @Description:
 */
public interface DataFatherMapper extends BaseMapper<DataFatherEntity> {


    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public DataFatherEntity selectDataFatherById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param dataFather 【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    public List<DataFatherEntity> selectDataFatherList(DataFatherEntity dataFather);

    /**
     * 新增【请填写功能名称】
     *
     * @param dataFather 【请填写功能名称】
     * @return 结果
     */
    public int insertDataFather(DataFatherEntity dataFather);

    /**
     * 修改【请填写功能名称】
     *
     * @param dataFather 【请填写功能名称】
     * @return 结果
     */
    public int updateDataFather(DataFatherEntity dataFather);

    /**
     * 删除【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteDataFatherById(Integer id);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteDataFatherByIds(String[] ids);


    IPage<DataResponse> SelectFatherResponseList(Page<DataResponse> pageParam, @Param("req") DataSonQueryRequest request);
    List<DataResponse> SelectFatherResponseListNoPage();
}
