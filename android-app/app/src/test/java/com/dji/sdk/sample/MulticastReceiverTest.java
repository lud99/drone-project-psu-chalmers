package com.dji.sdk.sample;

import org.json.JSONException;
import org.json.JSONObject;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.net.DatagramPacket;
import java.nio.charset.StandardCharsets;

public class MulticastReceiverTest {

    @Test
    public void testInvalidPacket()
    {
        MulticastReceiver rec = new MulticastReceiver(9992);

        byte[] buf = new byte[1024];
        buf[0] = '2';
        DatagramPacket packet = new DatagramPacket(buf, buf.length);
        assertFalse(rec.handlePacket(packet, buf));
    }

    @Test
    public void testInvalidJSON()
    {
        MulticastReceiver rec = new MulticastReceiver(9992);

        byte[] buf = new byte[1024];
        buf[0] = 'C';
        buf[1] = 'T';
        buf[2] = 'H';
        buf[3] = '{';
        DatagramPacket packet = new DatagramPacket(buf, buf.length);
        assertFalse(rec.handlePacket(packet, buf));
    }

    private byte[] constructMessage(String backendName, int port)
    {
        byte[] buf = new byte[MulticastReceiver.PayloadLength];
        buf[0] = 'C';
        buf[1] = 'T';
        buf[2] = 'H';

        JSONObject obj = new JSONObject();
        try {
            obj.put("msg_type", "backend_discovery");
            obj.put("name", backendName);
            obj.put("port", port);
            obj.put("ip", "192.168.0.1");

        } catch (JSONException e) {
            throw new RuntimeException(e);
        }

        byte[] jsonBytes = obj.toString().getBytes(StandardCharsets.UTF_8);

        if (jsonBytes.length + 3 > buf.length) {
            throw new IllegalArgumentException("Buffer too small for JSON payload");
        }

        System.arraycopy(jsonBytes, 0, buf, 3, jsonBytes.length);

        return buf;
    }

    @Test
    public void testValidMessage()
    {
        MulticastReceiver rec = new MulticastReceiver(9992);

        byte[] buf = constructMessage("Backend 1", 1234);
        DatagramPacket packet = new DatagramPacket(buf, buf.length);
        assertTrue(rec.handlePacket(packet, buf));

        buf = constructMessage("Backend 2", 9992);
        packet = new DatagramPacket(buf, buf.length);
        assertTrue(rec.handlePacket(packet, buf));

        buf = constructMessage("Backend 1", 8887);
        packet = new DatagramPacket(buf, buf.length);
        assertTrue(rec.handlePacket(packet, buf));

        // Assert it overrides
        assertEquals(2, rec.getAvailableBackends().size());
        assertEquals(8887, rec.getAvailableBackends().get(0).port);
    }
}
