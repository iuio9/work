package com.qczy.service.impl;

import com.qczy.model.entity.domain.Server;
import org.springframework.stereotype.Service;

@Service
public class HomeService {
    public Server getStoreageInfo(){
        Server server = new Server();
        try{
            server.copyTo();
        }catch (Exception e){
            e.printStackTrace();
        }

        return server;
    }
}
