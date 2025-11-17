package com.qczy.mapper;

import com.baomidou.mybatisplus.core.mapper.BaseMapper;
import com.qczy.model.entity.TempFileEntity;
import org.apache.ibatis.annotations.Delete;

import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/7 19:07
 * @Description:
 */
public interface TempFileMapper extends BaseMapper<TempFileEntity> {

    /**
     *  执行定时删除文件表数据
     */
    @Delete("TRUNCATE TABLE qczy_temp_file")
    public void deleteFileData();


    /**
     * 查询【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 【请填写功能名称】
     */
    public TempFileEntity selectTempFileById(Integer id);

    /**
     * 查询【请填写功能名称】列表
     *
     * @param tempFile 【请填写功能名称】
     * @return 【请填写功能名称】集合
     */
    public List<TempFileEntity> selectTempFileList(TempFileEntity tempFile);

    /**
     * 新增【请填写功能名称】
     *
     * @param tempFile 【请填写功能名称】
     * @return 结果
     */
    public int insertTempFile(TempFileEntity tempFile);

    /**
     * 修改【请填写功能名称】
     *
     * @param tempFile 【请填写功能名称】
     * @return 结果
     */
    public int updateTempFile(TempFileEntity tempFile);

    /**
     * 删除【请填写功能名称】
     *
     * @param id 【请填写功能名称】主键
     * @return 结果
     */
    public int deleteTempFileById(Integer id);

    /**
     * 批量删除【请填写功能名称】
     *
     * @param ids 需要删除的数据主键集合
     * @return 结果
     */
    public int deleteTempFileByIds(String[] ids);


}
