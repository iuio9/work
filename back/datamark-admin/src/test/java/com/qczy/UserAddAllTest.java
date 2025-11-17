package com.qczy;

import com.qczy.mapper.UserMapper;
import com.qczy.model.entity.UserEntity;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.junit4.SpringRunner;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/**
 * @Author: hh
 * @Version: 1.0
 * @Date: 2024/8/1 10:51
 * @Description:
 */
@SpringBootTest
@RunWith(SpringRunner.class)
public class UserAddAllTest {


/*
    @Autowired
    private UserMapper userMapper;

    @Test
    public void test() {


        for (int i = 1; i <= 50; i++) {
            UserEntity user = new UserEntity();
            user.setUserName("123" + i);
            user.setPassword("$2a$10$RiCnen.SlvjETfd1DR0bxeVqpCG//8rnGhPTbbZCEgkD9/gEnbp8y");
            user.setNickName("测试数据-" + i);
            user.setUserGender(i % 2 == 0 ? 1 : 2);
            user.setStatus(1);
            user.setUserPhone(i % 2 == 0 ?"18568572362":"15729449064");
            user.setUserEmail(i % 2 == 0 ? "1849499176@qq.com" : "15729449064@qq.com");
            user.setCreateTime(new Date());
            user.setUpdateTime(new Date());
            user.setIsDeleted(0);
            user.setUserRoles("1,2");
            userMapper.insert(user);
        }


    }
*/


    public static void main(String[] args) {
        int i = 2 / 4;
        System.out.println(i);
    }

}
